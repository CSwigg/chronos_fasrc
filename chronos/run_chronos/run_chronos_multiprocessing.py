#!/usr/bin/env python3
"""
Multiprocessing version of the Chronos cluster analysis notebook.
Processes multiple stellar clusters in parallel using multiprocessing.
"""

import argparse
from datetime import datetime
from pathlib import Path
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import arviz as az
import multiprocessing as mp
import time
from tqdm import tqdm
import warnings
import psutil  # For memory monitoring

# Suppress warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
plt.ioff()  # Turn off interactive plotting

from chronos.run_chronos.pipeline import (
    ChronosFitConfig,
    configure_cluster_fitter,
    mode_reals,
    prepare_member_photometry,
    select_clusters_for_chronos,
    summarize_skew_cauchy_samples,
)
from chronos.utils.ExtinctionPrior import ExtinctionPrior
from workflows.config import load_runtime_paths


def check_existing_results(output_file, all_results_file, cluster_ids):
    """
    Check existing results and return clusters that still need to be processed.

    Parameters:
    -----------
    output_file : str
        Path to the main output CSV file
    all_results_file : str
        Path to the all results CSV file
    cluster_ids : list
        List of all cluster IDs to process

    Returns:
    --------
    tuple
        (clusters_to_process, existing_results_df, existing_all_results_df)
    """
    existing_results_df = pd.DataFrame()
    existing_all_results_df = pd.DataFrame()

    # Check if results files exist
    if os.path.exists(all_results_file):
        try:
            existing_all_results_df = pd.read_csv(all_results_file)
            completed_clusters = set(existing_all_results_df['name'].tolist())
            print(f"Found existing results for {len(completed_clusters)} clusters")
        except Exception as e:
            print(f"Warning: Could not read existing results file: {e}")
            completed_clusters = set()
    else:
        completed_clusters = set()

    if os.path.exists(output_file):
        try:
            existing_results_df = pd.read_csv(output_file)
        except Exception as e:
            print(f"Warning: Could not read existing output file: {e}")

    # Determine which clusters still need processing
    all_clusters = set(cluster_ids)
    clusters_to_process = list(all_clusters - completed_clusters)

    if len(completed_clusters) > 0:
        print("Resuming analysis:")
        print(f"  Total clusters: {len(all_clusters)}")
        print(f"  Already completed: {len(completed_clusters)}")
        print(f"  Remaining to process: {len(clusters_to_process)}")
    else:
        print(f"Starting fresh analysis with {len(clusters_to_process)} clusters")

    return clusters_to_process, existing_results_df, existing_all_results_df


def save_results_batch(results_batch, df_clusters, output_file, all_results_file):
    """
    Save a batch of results to files efficiently.

    Parameters:
    -----------
    results_batch : list
        List of result dictionaries from processing clusters
    df_clusters : pd.DataFrame
        Original cluster data for merging
    output_file : str
        Path to main output file
    all_results_file : str
        Path to all results file
    """
    try:
        if not results_batch:
            return

        print(f"Saving batch of {len(results_batch)} results...")

        # Save to all results file
        if os.path.exists(all_results_file):
            all_results_df = pd.read_csv(all_results_file)
            # Remove test entries
            all_results_df = all_results_df[all_results_df['name'] != 'TEST_ENTRY']
        else:
            all_results_df = pd.DataFrame()

        # Add new results
        new_results_df = pd.DataFrame(results_batch)
        if len(all_results_df) > 0:
            # Remove any duplicates (in case of rerun)
            new_names = set(new_results_df['name'].tolist())
            all_results_df = all_results_df[~all_results_df['name'].isin(new_names)]
            all_results_df = pd.concat([all_results_df, new_results_df], ignore_index=True)
        else:
            all_results_df = new_results_df

        # Save all results
        all_results_df.to_csv(all_results_file, index=False)
        print(f"Saved all results to: {all_results_file}")

        # Save successful results with cluster data
        successful_results = [r for r in results_batch if r['status'] == 'success']
        if successful_results:
            print(f"Processing {len(successful_results)} successful results for main output...")

            # Load existing successful results
            if os.path.exists(output_file):
                existing_df = pd.read_csv(output_file)
                # Remove test entries
                existing_df = existing_df[existing_df['name'] != 'TEST_ENTRY']
            else:
                existing_df = pd.DataFrame()

            # Create DataFrame from successful results
            successful_df = pd.DataFrame(successful_results).drop('status', axis=1)

            # Get cluster data for merging (handle missing data_source column)
            if 'data_source' in df_clusters.columns:
                cluster_data = df_clusters[df_clusters['data_source'] == 'hunt'].copy()
            else:
                cluster_data = df_clusters.copy()

            # Filter to only clusters we have results for
            cluster_data = cluster_data[cluster_data['name'].isin(successful_df['name'])]

            # Merge cluster data with chronos results
            merged_df = pd.merge(cluster_data, successful_df, on='name', how='left')

            if len(existing_df) > 0:
                # Remove any existing entries for these clusters and combine
                existing_df = existing_df[~existing_df['name'].isin(merged_df['name'])]
                final_df = pd.concat([existing_df, merged_df], ignore_index=True)
            else:
                final_df = merged_df

            final_df.to_csv(output_file, index=False)
            print(f"Saved {len(successful_results)} successful results to: {output_file}")

        print("Batch save complete!")

    except Exception as e:
        print(f"Error saving batch: {e}")
        import traceback
        traceback.print_exc()


def process_cluster_simple(args):
    """
    Simple wrapper function that processes a cluster.
    This needs to be at module level for multiprocessing.
    """
    cluster_id, df, ext, output_dir_posterior, output_dir_isochrone, fit_config = args

    print(f"Processing cluster: {cluster_id}")
    result = process_single_cluster(
        cluster_id,
        df,
        ext,
        output_dir_posterior,
        output_dir_isochrone,
        fit_config=fit_config,
    )
    print(f"Finished cluster: {cluster_id} - Status: {result['status']}")

    return result


def process_single_cluster(cluster_id, data, ext, output_dir_posterior, output_dir_isochrone, *, fit_config):
    """
    Process a single cluster with Chronos Bayesian fitting.

    Parameters:
    -----------
    cluster_id : str
        Unique identifier for the cluster
    data : pd.DataFrame
        Full dataset containing all clusters
    ext : ExtinctionPrior
        Extinction prior object
    output_dir_posterior : str
        Directory to save posterior plots
    output_dir_isochrone : str
        Directory to save isochrone plots

    Returns:
    --------
    dict
        Dictionary containing the fitted parameters for this cluster
    """
    import gc  # Garbage collection for memory management

    try:
        # Get cluster data
        df_group = data.groupby('label').get_group(cluster_id)
        # Preserve the current Edenhofer query for auditability, even though the
        # production sampler still uses a flat A_V prior.
        _extinction_prior = ext.compute_prior(df_group['ra'], df_group['dec'], distance=df_group['distance_50'])

        cbayes = configure_cluster_fitter(df_group, fit_config)

        # Fit the model with reduced parameters to save memory
        _sampler, _best_fit, samples_bprp = cbayes.fit_bayesian(**fit_config.sampler_kwargs())
        posterior_summary = summarize_skew_cauchy_samples(
            samples_bprp,
            hdi_prob=fit_config.summary_hdi_prob,
        )

        # Get samples from posterior
        logAge, feh, A_V, skewness, scale = samples_bprp.T
        to_plot = 10**logAge / 10**6, A_V, skewness, scale
        names = 'Age (Myr)', 'AV (mag)', 'Skewness', 'Scale'

        # Create posterior plots
        fig, axes = plt.subplots(1, len(names), figsize=(len(names) * 4, 5))
        for name, data2plot, ax in zip(names, to_plot, axes):
            ax.hist(data2plot, bins=50, histtype='step', color='k')
            ax.hist(data2plot, bins=50, histtype='stepfilled', color='k', alpha=0.25)
            mode_hist = mode_reals(data2plot, bins=100)
            lo, hi = az.hdi(data2plot, hdi_prob=fit_config.posterior_plot_hdi_prob)
            for al in [mode_hist, lo, hi]:
                ax.axvline(al, c='k', alpha=0.5)
            ax.set_xlabel(name)

        posterior_file = os.path.join(output_dir_posterior, f'{cluster_id}_fit.png')
        plt.savefig(posterior_file, bbox_inches='tight', dpi=300)
        plt.close()
        plt.clf()

        age_mode = posterior_summary.age_mode
        age_lo = posterior_summary.age_lo
        age_hi = posterior_summary.age_hi
        av_mode = posterior_summary.av_mode
        av_lo = posterior_summary.av_lo
        av_hi = posterior_summary.av_hi

        # Compute fit info
        _, masses, _ = cbayes.compute_fit_info(
            logAge=np.log10(age_mode * 10**6), feh=0, A_V=av_mode, g_rp=cbayes.use_grp, signed_distance=True
        )

        # Create isochrone plot
        size = 15
        plt.figure(figsize=(6, 9))
        plt.scatter(*cbayes.distance_handler.fit_data['hrd'].T, s=50, c='tab:purple',
                    edgecolors='tab:purple', alpha=0.9)
        plt.ylim(14, -4)
        plt.xlim(-1, 5)
        plt.xlabel(r'$G_{BP} - G_{RP}$', size=size)
        plt.ylabel(r'$M_G$', size=size)
        plt.xticks(size=size)
        plt.yticks(size=size)

        # Plot isochrone
        isochrone = cbayes.isochrone_handler.model(
            logAge=np.log10(age_mode * 10**6), feh=0, A_V=av_mode, g_rp=cbayes.use_grp
        )
        plt.plot(*isochrone.T,
                 label=r'${{{:.1f}}}^{{+{:.1f}}}_{{{:.1f}}}$ Myr'.format(
                     age_mode, age_hi - age_mode, age_lo - age_mode),
                 c='k', alpha=0.7, zorder=0)
        plt.title(r'AV = ${{{:.1f}}}^{{{:.1f}}}_{{{:.1f}}}$ mag'.format(
            av_mode, av_hi - av_mode, av_lo - av_mode), size=size)
        plt.annotate(r'${{{:.1f}}}^{{+{:.1f}}}_{{{:.1f}}}$ Myr'.format(
            age_mode, age_hi - age_mode, age_lo - age_mode),
            (0.98, 0.98), xycoords='axes fraction', ha='right', va='top', size=size)

        isochrone_file = os.path.join(output_dir_isochrone, f'{cluster_id}.png')
        plt.savefig(isochrone_file, bbox_inches='tight', dpi=300)
        plt.close()
        plt.clf()

        # Force garbage collection to free memory
        gc.collect()

        # Return results
        result = {
            'name': cluster_id,
            'age_chronos_mode': age_mode,
            'age_chronos_lo': age_lo,
            'age_chronos_hi': age_hi,
            'av_chronos_mode': av_mode,
            'av_chronos_lo': av_lo,
            'av_chronos_hi': av_hi,
            'status': 'success'
        }

        return result

    except Exception as e:
        return {
            'name': cluster_id,
            'age_chronos_mode': np.nan,
            'age_chronos_lo': np.nan,
            'age_chronos_hi': np.nan,
            'av_chronos_mode': np.nan,
            'av_chronos_lo': np.nan,
            'av_chronos_hi': np.nan,
            'status': f'error: {str(e)}'
        }


def main(config_path: str | Path | None = None):
    """Main function to run the multiprocessing cluster analysis."""
    print("Starting Chronos multiprocessing cluster analysis...")
    print(f"Start time: {datetime.now()}")

    paths = load_runtime_paths(config_path)

    fit_config = ChronosFitConfig()

    # Define output files first
    output_file = paths.inputs.chronos_ages_csv
    all_results_file = paths.outputs.chronos_dir / "all_clusters_chronos_results.csv"
    paths.outputs.chronos_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading data...")
    df_stars = pd.read_csv(paths.inputs.member_catalog_csv)
    df_clusters = pd.read_csv(paths.inputs.cluster_catalog_csv)

    #### Make cuts
    df_clusters = select_clusters_for_chronos(
        df_clusters,
        max_input_age_myr=fit_config.max_input_age_myr,
    )
    print(f"Number of clusters after cuts: {len(df_clusters)}")

    # Process stellar data
    if 'distance_50' in df_stars.columns:
        print("Using 'distance_50' column for distances")
    else:
        print("Warning: 'distance_50' not found, using parallax-derived distances")
    df = prepare_member_photometry(df_stars, df_clusters)

    # Initialize extinction prior
    print("Loading extinction prior...")
    ext = ExtinctionPrior(str(paths.inputs.extinction_healpix_fits))

    # Create output directories
    output_dir_posterior = paths.outputs.chronos_dir / "posterior_plots"
    output_dir_isochrone = paths.outputs.chronos_dir / "isochrone_fit_plots"

    os.makedirs(output_dir_posterior, exist_ok=True)
    os.makedirs(output_dir_isochrone, exist_ok=True)

    # Get all cluster IDs and check what's already been processed
    all_cluster_ids = df.label.unique()
    clusters_to_process, existing_results_df, existing_all_results_df = check_existing_results(
        output_file, all_results_file, all_cluster_ids
    )

    if len(clusters_to_process) == 0:
        print("All clusters have already been processed!")
        print(f"Results are available in: {output_file}")
        return

    # Set up multiprocessing with safety limits
    n_processes = mp.cpu_count()
    print(f"Using {n_processes} processes")

    # Add memory management
    available_memory_gb = psutil.virtual_memory().available / (1024**3)
    print(f"Available memory: {available_memory_gb:.1f} GB")

    # Prepare arguments for multiprocessing
    process_args = [
        (cluster_id, df, ext, output_dir_posterior, output_dir_isochrone, fit_config)
        for cluster_id in clusters_to_process
    ]

    print("Starting cluster processing with batch saving every 5 clusters...")

    # Process clusters in batches
    batch_size = 5
    all_results = []

    # Process clusters in parallel
    start_time = time.time()

    try:
        with mp.Pool(processes=n_processes) as pool:
            for i, result in enumerate(tqdm(
                pool.imap(process_cluster_simple, process_args, chunksize=1),
                total=len(clusters_to_process),
                desc="Processing clusters",
                unit="cluster"
            )):
                all_results.append(result)

                # Save in batches of 5
                if (i + 1) % batch_size == 0:
                    batch_to_save = all_results[-batch_size:]
                    save_results_batch(batch_to_save, df_clusters, output_file, all_results_file)

                # Also save if we've reached the end
                elif (i + 1) == len(clusters_to_process):
                    remaining_batch = all_results[-(len(all_results) % batch_size):]
                    if remaining_batch:
                        save_results_batch(remaining_batch, df_clusters, output_file, all_results_file)

    except KeyboardInterrupt:
        print("\nKeyboard interrupt received. Saving current results...")
        if all_results:
            # Save whatever we have processed so far
            unsaved_start = len(all_results) - (len(all_results) % batch_size)
            if unsaved_start < len(all_results):
                remaining_batch = all_results[unsaved_start:]
                save_results_batch(remaining_batch, df_clusters, output_file, all_results_file)
            print(f"Saved {len(all_results)} processed results before exit")
    except Exception as e:
        print(f"\nError during processing: {e}")
        if all_results:
            save_results_batch(all_results, df_clusters, output_file, all_results_file)

    end_time = time.time()
    total_time = end_time - start_time

    print(f"\nProcessing complete! Total time: {total_time:.2f} seconds")

    # Optional: create summary
    try:
        if all_results:
            results_df = pd.DataFrame(all_results)
            successful_results = results_df[results_df['status'] == 'success']
            print(f"Summary: {len(successful_results)} clusters processed successfully out of {len(all_results)} total")

            if len(successful_results) > 0:
                print(f"Age range: {successful_results['age_chronos_mode'].min():.1f} - {successful_results['age_chronos_mode'].max():.1f} Myr")
                print(f"Extinction range: {successful_results['av_chronos_mode'].min():.3f} - {successful_results['av_chronos_mode'].max():.3f}")
        else:
            print("No results to summarize")
    except Exception as e:
        print(f"Could not create summary: {e}")

    print("Results saved to:")
    print(f"   - Main results: {output_file}")
    print(f"   - All results: {all_results_file}")
    print(f"   - Posterior plots: {output_dir_posterior}")
    print(f"   - Isochrone plots: {output_dir_isochrone}")

    # Force garbage collection
    import gc
    gc.collect()

    print("All done!")


if __name__ == "__main__":
    # Additional warning suppression for multiprocessing
    import logging

    logging.getLogger().setLevel(logging.ERROR)
    parser = argparse.ArgumentParser(description="Run Chronos age fitting with a shared supernova-map config.")
    parser.add_argument("--config", type=str, default=None, help="Path to workflow TOML config.")
    args = parser.parse_args()
    main(config_path=args.config)
