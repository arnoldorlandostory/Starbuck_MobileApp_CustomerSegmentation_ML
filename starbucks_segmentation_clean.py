"""
Starbucks Mobile App Market Segmentation Analysis
Author: Arnold Orlando
Description: Customer segmentation using RFM analysis and K-Means clustering
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Optional
from dataclasses import dataclass

from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

@dataclass
class Config:
    """Configuration constants for the analysis"""
    RANDOM_STATE: int = 42
    FIGSIZE_LARGE: Tuple[int, int] = (18, 5)
    FIGSIZE_MEDIUM: Tuple[int, int] = (15, 10)
    FIGSIZE_SMALL: Tuple[int, int] = (13, 5)
    
    # Column groups
    DISCRETE_COLS: List[str] = None
    CONTINUOUS_COLS: List[str] = None
    BOOLEAN_COLS: List[str] = None
    DROP_COLS: List[str] = None
    
    def __post_init__(self):
        self.DISCRETE_COLS = ['cart_size', 'num_customizations', 'customer_satisfaction']
        self.CONTINUOUS_COLS = ['total_spend', 'fulfillment_time_min']
        self.BOOLEAN_COLS = ['is_rewards_member', 'order_ahead']
        self.DROP_COLS = [
            'day_of_week', 'store_id', 'store_location_type', 'region',
            'customer_age_group', 'customer_gender', 'fulfillment_time_min',
            'drink_category', 'customer_satisfaction', 'cart_size',
            'num_customizations', 'has_food_item'
        ]


# ============================================================================
# DATA LOADING & PREPROCESSING
# ============================================================================

class DataLoader:
    """Handle data loading and initial preprocessing"""
    
    @staticmethod
    def load_data(filepath: str) -> pd.DataFrame:
        """
        Load data from CSV file
        
        Args:
            filepath: Path to the CSV file
            
        Returns:
            DataFrame containing the loaded data
        """
        try:
            df = pd.read_csv(filepath)
            print(f"✓ Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {filepath}")
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")
    
    @staticmethod
    def filter_mobile_app(df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter data for Mobile App orders only
        
        Args:
            df: Input DataFrame
            
        Returns:
            Filtered DataFrame
        """
        mobile_df = df[df['order_channel'] == 'Mobile App'].copy()
        print(f"✓ Filtered Mobile App orders: {mobile_df.shape[0]} rows")
        return mobile_df
    
    @staticmethod
    def convert_date_column(df: pd.DataFrame, date_col: str = 'order_date') -> pd.DataFrame:
        """
        Convert date column to datetime format
        
        Args:
            df: Input DataFrame
            date_col: Name of the date column
            
        Returns:
            DataFrame with converted date column
        """
        df[date_col] = pd.to_datetime(df[date_col], format='%d/%m/%Y')
        print(f"✓ Date column '{date_col}' converted to datetime")
        return df


# ============================================================================
# EXPLORATORY DATA ANALYSIS
# ============================================================================

class EDA:
    """Exploratory Data Analysis utilities"""
    
    @staticmethod
    def print_data_summary(df: pd.DataFrame):
        """Print comprehensive data summary"""
        print("\n" + "="*70)
        print("DATA SUMMARY")
        print("="*70)
        print(f"\nShape: {df.shape}")
        print(f"\nMissing values:\n{df.isna().sum()}")
        print(f"\nZero values:\n{(df == 0).sum()}")
        print(f"\nDuplicates: {df.duplicated().sum()}")
        print("\n" + "="*70 + "\n")
    
    @staticmethod
    def plot_continuous_distributions(df: pd.DataFrame, columns: List[str], 
                                     figsize: Tuple[int, int] = (13, 5)):
        """
        Plot KDE distributions for continuous variables
        
        Args:
            df: Input DataFrame
            columns: List of column names to plot
            figsize: Figure size
        """
        n_cols = len(columns)
        fig, axes = plt.subplots(1, n_cols, figsize=figsize)
        axes = axes.flatten() if n_cols > 1 else [axes]
        
        for i, col in enumerate(columns):
            sns.kdeplot(data=df[col], ax=axes[i], fill=True)
            axes[i].set_title(f'Distribution of {col}', fontsize=12, fontweight='bold')
            axes[i].set_xlabel(col)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_discrete_distributions(df: pd.DataFrame, columns: List[str],
                                   figsize: Tuple[int, int] = (18, 5)):
        """
        Plot histograms for discrete variables
        
        Args:
            df: Input DataFrame
            columns: List of column names to plot
            figsize: Figure size
        """
        n_cols = len(columns)
        fig, axes = plt.subplots(1, n_cols, figsize=figsize)
        axes = axes.flatten() if n_cols > 1 else [axes]
        
        for i, col in enumerate(columns):
            sns.histplot(data=df[col], ax=axes[i], kde=False, bins=20)
            axes[i].set_title(f'Distribution of {col}', fontsize=12, fontweight='bold')
            axes[i].set_xlabel(col)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_correlation_matrix(df: pd.DataFrame, figsize: Tuple[int, int] = (10, 8)):
        """
        Plot correlation heatmap
        
        Args:
            df: Input DataFrame
            figsize: Figure size
        """
        plt.figure(figsize=figsize)
        corr_matrix = df.corr(numeric_only=True)
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, linewidths=1)
        plt.title('Correlation Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_boxplots(df: pd.DataFrame, columns: List[str],
                     figsize: Tuple[int, int] = (15, 10)):
        """
        Plot boxplots to detect outliers
        
        Args:
            df: Input DataFrame
            columns: List of column names to plot
            figsize: Figure size
        """
        n_cols = 3
        n_rows = (len(columns) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten()
        
        for i, col in enumerate(columns):
            sns.boxplot(x=df[col], ax=axes[i])
            axes[i].set_title(f'Boxplot of {col}', fontsize=11, fontweight='bold')
        
        # Hide empty subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')
        
        plt.tight_layout()
        plt.show()


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

class FeatureEngineering:
    """Feature engineering and transformation utilities"""
    
    @staticmethod
    def encode_ordinal(df: pd.DataFrame, columns: List[str]) -> Tuple[pd.DataFrame, OrdinalEncoder]:
        """
        Apply ordinal encoding to specified columns
        
        Args:
            df: Input DataFrame
            columns: List of columns to encode
            
        Returns:
            Tuple of (encoded DataFrame, fitted encoder)
        """
        encoder = OrdinalEncoder(categories='auto')
        data_subset = df[columns]
        encoded_data = encoder.fit_transform(data_subset)
        encoded_df = pd.DataFrame(encoded_data, columns=columns, index=df.index)
        
        print(f"✓ Ordinal encoding applied to: {', '.join(columns)}")
        return encoded_df, encoder
    
    @staticmethod
    def create_rfm_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Create RFM (Recency, Frequency, Monetary) features
        
        Args:
            df: Input DataFrame with customer transactions
            
        Returns:
            DataFrame with RFM features
        """
        # Aggregate by customer
        customer_agg = df.groupby('customer_id').agg({
            'order_id': 'count',
            'order_date': 'max',
            'total_spend': 'sum',
            'is_rewards_member': 'mean',
            'order_ahead': 'mean'
        }).reset_index()
        
        customer_agg.columns = ['customer_id', 'order_count', 'last_order_date',
                               'total_spend', 'rewards_member', 'order_ahead']
        
        # Calculate Recency
        max_date = customer_agg['last_order_date'].max()
        customer_agg['recency'] = (max_date - customer_agg['last_order_date']).dt.days
        
        # Select final features
        rfm_df = customer_agg[[
            'recency', 'order_count', 'total_spend', 
            'rewards_member', 'order_ahead'
        ]].copy()
        
        rfm_df.columns = ['recency', 'frequency', 'monetary', 
                         'rewards_member', 'order_ahead']
        
        print(f"✓ RFM features created: {rfm_df.shape}")
        return rfm_df
    
    @staticmethod
    def scale_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, StandardScaler]:
        """
        Standardize features using StandardScaler
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (scaled DataFrame, fitted scaler)
        """
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df)
        scaled_df = pd.DataFrame(scaled_data, columns=df.columns, index=df.index)
        
        print(f"✓ Features scaled using StandardScaler")
        return scaled_df, scaler
    
    @staticmethod
    def apply_pca(df: pd.DataFrame, n_components: Optional[int] = None, 
                 variance_threshold: float = 0.95) -> Tuple[pd.DataFrame, PCA]:
        """
        Apply PCA for dimensionality reduction
        
        Args:
            df: Input DataFrame
            n_components: Number of components (if None, auto-select based on variance)
            variance_threshold: Cumulative variance threshold for auto-selection
            
        Returns:
            Tuple of (PCA-transformed DataFrame, fitted PCA object)
        """
        # If n_components not specified, find optimal number
        if n_components is None:
            pca_temp = PCA(random_state=42)
            pca_temp.fit(df)
            
            cumulative_variance = np.cumsum(pca_temp.explained_variance_ratio_)
            n_components = np.argmax(cumulative_variance >= variance_threshold) + 1
            
            print("\nPCA Variance Analysis:")
            print("-" * 50)
            for i in range(len(cumulative_variance)):
                print(f"  PC {i+1}: {cumulative_variance[i]*100:.2f}% cumulative variance")
            print(f"\n✓ Selected {n_components} components (≥{variance_threshold*100}% variance)")
        
        # Apply PCA with selected components
        pca = PCA(n_components=n_components, random_state=42)
        pca_data = pca.fit_transform(df)
        
        # Create DataFrame with proper column names
        pca_columns = [f'PC_{i+1}' for i in range(n_components)]
        pca_df = pd.DataFrame(pca_data, columns=pca_columns, index=df.index)
        
        return pca_df, pca


# ============================================================================
# CLUSTERING
# ============================================================================

class ClusteringAnalysis:
    """K-Means clustering and evaluation utilities"""
    
    @staticmethod
    def find_optimal_clusters_elbow(df: pd.DataFrame, k_range: np.ndarray = None,
                                   plot: bool = True) -> pd.DataFrame:
        """
        Use Elbow method to find optimal number of clusters
        
        Args:
            df: Input DataFrame
            k_range: Range of k values to test
            plot: Whether to plot the elbow curve
            
        Returns:
            DataFrame with WSS values for each k
        """
        if k_range is None:
            k_range = np.arange(1, 12)
        
        wss_values = []
        for k in k_range:
            kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
            kmeans.fit(df)
            wss_values.append(kmeans.inertia_)
        
        wss_df = pd.DataFrame({'n_clusters': k_range, 'WSS': wss_values})
        
        if plot:
            plt.figure(figsize=(10, 6))
            sns.lineplot(data=wss_df, x='n_clusters', y='WSS', marker='o', linewidth=2)
            plt.title('Elbow Method - Within-Cluster Sum of Squares', 
                     fontsize=14, fontweight='bold')
            plt.xlabel('Number of Clusters', fontsize=12)
            plt.ylabel('WSS', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
        
        return wss_df
    
    @staticmethod
    def find_optimal_clusters_silhouette(df: pd.DataFrame, k_range: np.ndarray = None,
                                        plot: bool = True) -> Tuple[pd.DataFrame, int]:
        """
        Use Silhouette method to find optimal number of clusters
        
        Args:
            df: Input DataFrame
            k_range: Range of k values to test
            plot: Whether to plot the silhouette scores
            
        Returns:
            Tuple of (DataFrame with silhouette scores, optimal k)
        """
        if k_range is None:
            k_range = np.arange(2, 11)
        
        silhouette_scores = []
        print("\nSilhouette Score Analysis:")
        print("-" * 50)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
            labels = kmeans.fit_predict(df)
            score = silhouette_score(df, labels, metric='euclidean', 
                                   sample_size=min(1000, len(df)), random_state=42)
            silhouette_scores.append(score)
            print(f"  k={k}: Silhouette Score = {score:.4f}")
        
        silhouette_df = pd.DataFrame({
            'n_clusters': k_range, 
            'silhouette_score': silhouette_scores
        })
        
        optimal_k = k_range[np.argmax(silhouette_scores)]
        print(f"\n✓ Optimal clusters: {optimal_k} (highest silhouette score)")
        
        if plot:
            plt.figure(figsize=(10, 6))
            sns.lineplot(data=silhouette_df, x='n_clusters', y='silhouette_score', 
                        marker='o', linewidth=2)
            plt.axvline(optimal_k, color='red', linestyle='--', alpha=0.7, 
                       label=f'Optimal k={optimal_k}')
            plt.title('Silhouette Analysis', fontsize=14, fontweight='bold')
            plt.xlabel('Number of Clusters', fontsize=12)
            plt.ylabel('Silhouette Score', fontsize=12)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
        
        return silhouette_df, optimal_k
    
    @staticmethod
    def perform_clustering(df: pd.DataFrame, n_clusters: int) -> Tuple[KMeans, np.ndarray]:
        """
        Perform K-Means clustering
        
        Args:
            df: Input DataFrame
            n_clusters: Number of clusters
            
        Returns:
            Tuple of (fitted KMeans model, cluster labels)
        """
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', 
                       random_state=42, n_init=10)
        labels = kmeans.fit_predict(df)
        
        print(f"\n✓ K-Means clustering completed with {n_clusters} clusters")
        print(f"  Cluster distribution: {np.bincount(labels)}")
        
        return kmeans, labels
    
    @staticmethod
    def get_cluster_centers(kmeans: KMeans, pca: PCA, scaler: StandardScaler,
                          feature_names: List[str]) -> pd.DataFrame:
        """
        Get cluster centers in original feature space
        
        Args:
            kmeans: Fitted KMeans model
            pca: Fitted PCA object
            scaler: Fitted StandardScaler object
            feature_names: Original feature names
            
        Returns:
            DataFrame with cluster centers
        """
        # Get centers in PCA space
        centers_pca = kmeans.cluster_centers_
        
        # Inverse transform from PCA space to scaled space
        centers_scaled = pca.inverse_transform(centers_pca)
        
        # Inverse transform from scaled space to original space
        centers_original = scaler.inverse_transform(centers_scaled)
        
        # Create DataFrame
        centers_df = pd.DataFrame(centers_original, columns=feature_names)
        centers_df.insert(0, 'cluster', [f'Cluster {i+1}' for i in range(len(centers_df))])
        
        return centers_df


# ============================================================================
# MAIN PIPELINE
# ============================================================================

class StarbucksSegmentation:
    """Main pipeline for Starbucks customer segmentation"""
    
    def __init__(self, filepath: str):
        """
        Initialize the segmentation pipeline
        
        Args:
            filepath: Path to the data file
        """
        self.config = Config()
        self.filepath = filepath
        self.data = None
        self.rfm_data = None
        self.scaled_data = None
        self.pca_data = None
        self.cluster_centers = None
        
        # Store fitted transformers
        self.scaler = None
        self.pca = None
        self.kmeans = None
    
    def load_and_prepare_data(self):
        """Load and prepare data for analysis"""
        print("\n" + "="*70)
        print("STEP 1: DATA LOADING & PREPARATION")
        print("="*70)
        
        # Load data
        self.data = DataLoader.load_data(self.filepath)
        
        # Convert date
        self.data = DataLoader.convert_date_column(self.data)
        
        # Drop unnecessary columns
        self.data.drop(columns=self.config.DROP_COLS, inplace=True, errors='ignore')
        
        # Filter Mobile App only
        self.data = DataLoader.filter_mobile_app(self.data)
        
        print("✓ Data preparation completed\n")
    
    def explore_data(self):
        """Perform exploratory data analysis"""
        print("\n" + "="*70)
        print("STEP 2: EXPLORATORY DATA ANALYSIS")
        print("="*70)
        
        EDA.print_data_summary(self.data)
        
        # Plot distributions (optional - uncomment to visualize)
        # EDA.plot_continuous_distributions(self.data, self.config.CONTINUOUS_COLS)
        # EDA.plot_discrete_distributions(self.data, self.config.DISCRETE_COLS)
        # EDA.plot_correlation_matrix(self.data)
        # EDA.plot_boxplots(self.data, self.config.CONTINUOUS_COLS + self.config.DISCRETE_COLS)
    
    def engineer_features(self):
        """Engineer RFM and other features"""
        print("\n" + "="*70)
        print("STEP 3: FEATURE ENGINEERING")
        print("="*70)
        
        # Encode boolean columns
        encoded_data, _ = FeatureEngineering.encode_ordinal(
            self.data, self.config.BOOLEAN_COLS
        )
        
        # Combine with original data
        data_combined = pd.concat([
            self.data.drop(columns=self.config.BOOLEAN_COLS),
            encoded_data
        ], axis=1)
        
        # Create RFM features
        self.rfm_data = FeatureEngineering.create_rfm_features(data_combined)
        
        print("✓ Feature engineering completed\n")
    
    def scale_and_reduce(self, n_components: Optional[int] = None):
        """Scale features and apply PCA"""
        print("\n" + "="*70)
        print("STEP 4: FEATURE SCALING & DIMENSIONALITY REDUCTION")
        print("="*70)
        
        # Scale features
        self.scaled_data, self.scaler = FeatureEngineering.scale_features(self.rfm_data)
        
        # Apply PCA
        self.pca_data, self.pca = FeatureEngineering.apply_pca(
            self.scaled_data, n_components=n_components
        )
        
        print("✓ Scaling and PCA completed\n")
    
    def find_optimal_clusters(self):
        """Find optimal number of clusters"""
        print("\n" + "="*70)
        print("STEP 5: FINDING OPTIMAL NUMBER OF CLUSTERS")
        print("="*70)
        
        # Elbow method
        print("\n--- Elbow Method ---")
        wss_df = ClusteringAnalysis.find_optimal_clusters_elbow(self.pca_data)
        
        # Silhouette method
        print("\n--- Silhouette Method ---")
        silhouette_df, optimal_k = ClusteringAnalysis.find_optimal_clusters_silhouette(
            self.pca_data
        )
        
        return optimal_k
    
    def perform_clustering(self, n_clusters: int):
        """Perform final clustering"""
        print("\n" + "="*70)
        print("STEP 6: FINAL CLUSTERING")
        print("="*70)
        
        # Perform clustering
        self.kmeans, labels = ClusteringAnalysis.perform_clustering(
            self.pca_data, n_clusters
        )
        
        # Get cluster centers
        self.cluster_centers = ClusteringAnalysis.get_cluster_centers(
            self.kmeans, self.pca, self.scaler, self.rfm_data.columns.tolist()
        )
        
        print("\nCluster Centers (Original Scale):")
        print(self.cluster_centers.to_string(index=False))
        print("\n✓ Clustering completed\n")
        
        return self.cluster_centers
    
    def run_pipeline(self, auto_select_clusters: bool = True, n_clusters: Optional[int] = None):
        """
        Run the complete segmentation pipeline
        
        Args:
            auto_select_clusters: Whether to auto-select optimal clusters
            n_clusters: Manual number of clusters (used if auto_select_clusters=False)
            
        Returns:
            DataFrame with cluster centers
        """
        # Step 1: Load and prepare data
        self.load_and_prepare_data()
        
        # Step 2: Explore data
        self.explore_data()
        
        # Step 3: Engineer features
        self.engineer_features()
        
        # Step 4: Scale and reduce dimensions
        self.scale_and_reduce()
        
        # Step 5: Find optimal clusters
        if auto_select_clusters:
            optimal_k = self.find_optimal_clusters()
        else:
            optimal_k = n_clusters
        
        # Step 6: Perform final clustering
        cluster_centers = self.perform_clustering(optimal_k)
        
        print("\n" + "="*70)
        print("SEGMENTATION PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*70 + "\n")
        
        return cluster_centers


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Initialize and run pipeline
    segmentation = StarbucksSegmentation('starbucks_customer_ordering_patterns.csv')
    
    # Run complete pipeline with automatic cluster selection
    cluster_centers = segmentation.run_pipeline(auto_select_clusters=True)
    
    # Access results
    print("\nFinal Cluster Centers:")
    print(cluster_centers)
    
    # Optional: Save results
    # cluster_centers.to_csv('cluster_centers.csv', index=False)
