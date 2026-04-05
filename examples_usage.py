"""
Example Usage - Starbucks Market Segmentation Clean Code
=========================================================

This file demonstrates various ways to use the clean code implementation
"""

from starbucks_segmentation_clean import (
    StarbucksSegmentation,
    DataLoader,
    EDA,
    FeatureEngineering,
    ClusteringAnalysis,
    Config
)
import pandas as pd


# ============================================================================
# EXAMPLE 1: Quick Start - Full Pipeline
# ============================================================================

def example_1_quick_start():
    """Run the complete pipeline with default settings"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Quick Start - Full Pipeline")
    print("="*70)
    
    segmentation = StarbucksSegmentation('starbucks_customer_ordering_patterns.csv')
    cluster_centers = segmentation.run_pipeline(auto_select_clusters=True)
    
    # Save results
    cluster_centers.to_csv('cluster_results.csv', index=False)
    print("\n✓ Results saved to 'cluster_results.csv'")


# ============================================================================
# EXAMPLE 2: Manual Cluster Selection
# ============================================================================

def example_2_manual_clusters():
    """Run pipeline with manually specified number of clusters"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Manual Cluster Selection")
    print("="*70)
    
    segmentation = StarbucksSegmentation('starbucks_customer_ordering_patterns.csv')
    
    # Specify 4 clusters manually
    cluster_centers = segmentation.run_pipeline(
        auto_select_clusters=False, 
        n_clusters=4
    )
    
    return cluster_centers


# ============================================================================
# EXAMPLE 3: Step-by-Step Execution with Custom Analysis
# ============================================================================

def example_3_step_by_step():
    """Run pipeline step by step with custom analysis between steps"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Step-by-Step Execution")
    print("="*70)
    
    segmentation = StarbucksSegmentation('starbucks_customer_ordering_patterns.csv')
    
    # Step 1: Load and prepare
    segmentation.load_and_prepare_data()
    print(f"Data shape: {segmentation.data.shape}")
    
    # Step 2: EDA
    segmentation.explore_data()
    
    # Custom analysis: Check rewards member distribution
    rewards_pct = segmentation.data['is_rewards_member'].mean() * 100
    print(f"\nRewards members: {rewards_pct:.1f}%")
    
    # Step 3: Feature engineering
    segmentation.engineer_features()
    print(f"RFM data shape: {segmentation.rfm_data.shape}")
    
    # Step 4: Scaling and PCA
    segmentation.scale_and_reduce(n_components=3)
    print(f"PCA data shape: {segmentation.pca_data.shape}")
    
    # Step 5: Find optimal clusters
    optimal_k = segmentation.find_optimal_clusters()
    
    # Step 6: Clustering
    cluster_centers = segmentation.perform_clustering(optimal_k)
    
    return cluster_centers


# ============================================================================
# EXAMPLE 4: Using Individual Components
# ============================================================================

def example_4_individual_components():
    """Use individual components separately"""
    print("\n" + "="*70)
    print("EXAMPLE 4: Individual Components")
    print("="*70)
    
    # Load data
    df = DataLoader.load_data('starbucks_customer_ordering_patterns.csv')
    
    # Filter Mobile App
    df_mobile = DataLoader.filter_mobile_app(df)
    
    # Convert dates
    df_mobile = DataLoader.convert_date_column(df_mobile)
    
    # Print summary
    EDA.print_data_summary(df_mobile)
    
    # Plot correlation (commented out to avoid GUI in example)
    # EDA.plot_correlation_matrix(df_mobile)
    
    return df_mobile


# ============================================================================
# EXAMPLE 5: Custom Configuration
# ============================================================================

def example_5_custom_config():
    """Use custom configuration"""
    print("\n" + "="*70)
    print("EXAMPLE 5: Custom Configuration")
    print("="*70)
    
    # Create custom config
    config = Config()
    config.RANDOM_STATE = 123
    
    # Load data
    segmentation = StarbucksSegmentation('starbucks_customer_ordering_patterns.csv')
    segmentation.config = config  # Apply custom config
    
    # Run pipeline
    cluster_centers = segmentation.run_pipeline(auto_select_clusters=True)
    
    return cluster_centers


# ============================================================================
# EXAMPLE 6: Analyzing Cluster Characteristics
# ============================================================================

def example_6_analyze_clusters():
    """Run clustering and analyze cluster characteristics"""
    print("\n" + "="*70)
    print("EXAMPLE 6: Cluster Analysis")
    print("="*70)
    
    segmentation = StarbucksSegmentation('starbucks_customer_ordering_patterns.csv')
    cluster_centers = segmentation.run_pipeline(auto_select_clusters=True)
    
    # Analyze each cluster
    print("\n" + "="*70)
    print("CLUSTER CHARACTERISTICS")
    print("="*70)
    
    for idx, row in cluster_centers.iterrows():
        print(f"\n{row['cluster']}:")
        print(f"  - Recency: {row['recency']:.1f} days")
        print(f"  - Frequency: {row['frequency']:.1f} orders")
        print(f"  - Monetary: ${row['monetary']:.2f}")
        print(f"  - Rewards Member: {row['rewards_member']*100:.1f}%")
        print(f"  - Order Ahead: {row['order_ahead']*100:.1f}%")
        
        # Interpret cluster
        if row['frequency'] > cluster_centers['frequency'].mean():
            print(f"  → High-frequency customers")
        if row['monetary'] > cluster_centers['monetary'].mean():
            print(f"  → High-value customers")
        if row['recency'] < cluster_centers['recency'].mean():
            print(f"  → Recently active")
    
    return cluster_centers


# ============================================================================
# EXAMPLE 7: Comparing Different PCA Components
# ============================================================================

def example_7_compare_pca():
    """Compare clustering results with different PCA components"""
    print("\n" + "="*70)
    print("EXAMPLE 7: PCA Component Comparison")
    print("="*70)
    
    segmentation = StarbucksSegmentation('starbucks_customer_ordering_patterns.csv')
    
    # Prepare data
    segmentation.load_and_prepare_data()
    segmentation.engineer_features()
    
    results = {}
    
    # Try different numbers of components
    for n_comp in [2, 3, 4, 5]:
        print(f"\n--- Testing {n_comp} PCA components ---")
        
        segmentation.scale_and_reduce(n_components=n_comp)
        
        # Find optimal clusters
        _, optimal_k = ClusteringAnalysis.find_optimal_clusters_silhouette(
            segmentation.pca_data, plot=False
        )
        
        results[n_comp] = optimal_k
        print(f"Optimal clusters: {optimal_k}")
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for n_comp, k in results.items():
        print(f"{n_comp} components → {k} clusters")
    
    return results


# ============================================================================
# EXAMPLE 8: Export and Visualization
# ============================================================================

def example_8_export_results():
    """Run analysis and export comprehensive results"""
    print("\n" + "="*70)
    print("EXAMPLE 8: Export Comprehensive Results")
    print("="*70)
    
    segmentation = StarbucksSegmentation('starbucks_customer_ordering_patterns.csv')
    cluster_centers = segmentation.run_pipeline(auto_select_clusters=True)
    
    # Export cluster centers
    cluster_centers.to_csv('output_cluster_centers.csv', index=False)
    print("✓ Cluster centers saved to 'output_cluster_centers.csv'")
    
    # Export RFM data with cluster labels
    rfm_with_labels = segmentation.rfm_data.copy()
    rfm_with_labels['cluster'] = segmentation.kmeans.labels_
    rfm_with_labels.to_csv('output_rfm_with_clusters.csv', index=False)
    print("✓ RFM data with clusters saved to 'output_rfm_with_clusters.csv'")
    
    # Export PCA data
    pca_with_labels = segmentation.pca_data.copy()
    pca_with_labels['cluster'] = segmentation.kmeans.labels_
    pca_with_labels.to_csv('output_pca_with_clusters.csv', index=False)
    print("✓ PCA data with clusters saved to 'output_pca_with_clusters.csv'")
    
    # Create summary statistics
    summary = {
        'total_customers': len(segmentation.rfm_data),
        'n_clusters': len(cluster_centers),
        'pca_components': segmentation.pca_data.shape[1],
        'variance_explained': sum(segmentation.pca.explained_variance_ratio_) * 100
    }
    
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv('output_summary.csv', index=False)
    print("✓ Summary statistics saved to 'output_summary.csv'")
    
    return cluster_centers


# ============================================================================
# MAIN - Run Examples
# ============================================================================

if __name__ == "__main__":
    # Choose which example to run
    import sys
    
    examples = {
        '1': ('Quick Start - Full Pipeline', example_1_quick_start),
        '2': ('Manual Cluster Selection', example_2_manual_clusters),
        '3': ('Step-by-Step Execution', example_3_step_by_step),
        '4': ('Individual Components', example_4_individual_components),
        '5': ('Custom Configuration', example_5_custom_config),
        '6': ('Cluster Analysis', example_6_analyze_clusters),
        '7': ('PCA Component Comparison', example_7_compare_pca),
        '8': ('Export Results', example_8_export_results),
    }
    
    print("\n" + "="*70)
    print("STARBUCKS SEGMENTATION - EXAMPLE USAGE")
    print("="*70)
    print("\nAvailable Examples:")
    for key, (name, _) in examples.items():
        print(f"  {key}. {name}")
    
    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        choice = input("\nEnter example number (or 'all'): ").strip()
    
    if choice.lower() == 'all':
        for key, (name, func) in examples.items():
            try:
                func()
            except Exception as e:
                print(f"\n⚠ Error in Example {key}: {str(e)}")
    elif choice in examples:
        _, func = examples[choice]
        func()
    else:
        print("Invalid choice. Running Example 1 (Quick Start)...")
        example_1_quick_start()
    
    print("\n" + "="*70)
    print("EXAMPLES COMPLETED")
    print("="*70 + "\n")
