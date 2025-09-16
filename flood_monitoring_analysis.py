#!/usr/bin/env python3
"""
Flood Monitoring Analysis with Threshold-Based Risk Assessment
This script analyzes flood data and creates visualizations with color-coded risk levels.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os

def load_and_prepare_data(csv_file='Flood_Data.csv'):
    """Load and prepare the flood data for analysis."""
    print("Loading flood data...")
    
    # Load the flood data
    df = pd.read_csv(csv_file)
    
    # Clean column names (remove extra spaces)
    df.columns = df.columns.str.strip()
    
    print(f"Data loaded successfully. Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    return df

def analyze_threshold(df, threshold=51.4):
    """Analyze data based on threshold and create risk labels."""
    print(f"\nAnalyzing data with threshold: {threshold}")
    
    # Create status labels based on threshold
    df['Status'] = df['Distance'].apply(lambda x: 'High Risk (Red)' if x > threshold else 'Safe (Green)')
    df['Color'] = df['Distance'].apply(lambda x: 'red' if x > threshold else 'green')
    df['Risk_Level'] = df['Distance'].apply(lambda x: 'RED' if x > threshold else 'GREEN')
    
    # Calculate statistics
    total_records = len(df)
    high_risk_count = len(df[df['Distance'] > threshold])
    safe_count = len(df[df['Distance'] <= threshold])
    
    print(f"Total records: {total_records}")
    print(f"Records above threshold (High Risk - Red): {high_risk_count} ({high_risk_count/total_records*100:.1f}%)")
    print(f"Records below/at threshold (Safe - Green): {safe_count} ({safe_count/total_records*100:.1f}%)")
    
    return df, threshold

def create_comprehensive_visualization(df, threshold):
    """Create a comprehensive visualization with multiple plots."""
    print("\nCreating comprehensive visualization...")
    
    # Set up the figure with subplots
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('Flood Monitoring Analysis with Threshold-Based Risk Assessment', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Time series scatter plot with color coding
    ax1 = plt.subplot(2, 3, 1)
    colors = ['red' if x > threshold else 'green' for x in df['Distance']]
    scatter = plt.scatter(range(len(df)), df['Distance'], c=colors, alpha=0.7, s=30)
    plt.axhline(y=threshold, color='orange', linestyle='--', linewidth=2, 
                label=f'Threshold ({threshold})')
    plt.xlabel('Data Point Index')
    plt.ylabel('Water Level (Distance)')
    plt.title('Water Level with Threshold Line and Color Coding')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Histogram with threshold line
    ax2 = plt.subplot(2, 3, 2)
    safe_data = df[df['Distance'] <= threshold]['Distance']
    risk_data = df[df['Distance'] > threshold]['Distance']
    
    plt.hist(safe_data, bins=15, alpha=0.7, color='green', label='Safe (Green)', 
             edgecolor='black', density=False)
    if len(risk_data) > 0:
        plt.hist(risk_data, bins=15, alpha=0.7, color='red', label='High Risk (Red)', 
                 edgecolor='black', density=False)
    
    plt.axvline(x=threshold, color='orange', linestyle='--', linewidth=2, 
                label=f'Threshold ({threshold})')
    plt.xlabel('Water Level (Distance)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Water Levels')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Status distribution pie chart
    ax3 = plt.subplot(2, 3, 3)
    status_counts = df['Status'].value_counts()
    colors_pie = ['green' if 'Green' in label else 'red' for label in status_counts.index]
    wedges, texts, autotexts = plt.pie(status_counts.values, labels=status_counts.index, 
                                       autopct='%1.1f%%', colors=colors_pie, startangle=90)
    plt.title('Risk Status Distribution')
    
    # Plot 4: Box plot with threshold line
    ax4 = plt.subplot(2, 3, 4)
    if len(risk_data) > 0:
        box_data = [safe_data, risk_data]
        box_labels = ['Safe (≤ Threshold)', 'High Risk (> Threshold)']
    else:
        box_data = [safe_data]
        box_labels = ['Safe (≤ Threshold)']
    
    bp = plt.boxplot(box_data, labels=box_labels, patch_artist=True)
    bp['boxes'][0].set_facecolor('green')
    bp['boxes'][0].set_alpha(0.7)
    if len(bp['boxes']) > 1:
        bp['boxes'][1].set_facecolor('red')
        bp['boxes'][1].set_alpha(0.7)
    
    plt.axhline(y=threshold, color='orange', linestyle='--', linewidth=2, 
                label=f'Threshold ({threshold})')
    plt.ylabel('Water Level (Distance)')
    plt.title('Water Level Distribution by Risk Status')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Risk level over data points (line plot)
    ax5 = plt.subplot(2, 3, 5)
    plt.plot(range(len(df)), df['Distance'], color='blue', alpha=0.6, linewidth=1)
    plt.fill_between(range(len(df)), df['Distance'], threshold, 
                     where=(df['Distance'] > threshold), color='red', alpha=0.3, 
                     label='High Risk Area')
    plt.fill_between(range(len(df)), df['Distance'], threshold, 
                     where=(df['Distance'] <= threshold), color='green', alpha=0.3, 
                     label='Safe Area')
    plt.axhline(y=threshold, color='orange', linestyle='--', linewidth=2, 
                label=f'Threshold ({threshold})')
    plt.xlabel('Data Point Index')
    plt.ylabel('Water Level (Distance)')
    plt.title('Water Level Trend with Risk Areas')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 6: Statistics summary (text plot)
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    # Calculate statistics
    stats_text = f"""
FLOOD MONITORING STATISTICS

Threshold Value: {threshold}
Total Measurements: {len(df)}

High Risk (Red): {len(df[df['Distance'] > threshold])} records
({len(df[df['Distance'] > threshold])/len(df)*100:.1f}%)

Safe (Green): {len(df[df['Distance'] <= threshold])} records
({len(df[df['Distance'] <= threshold])/len(df)*100:.1f}%)

Water Level Statistics:
• Average: {df['Distance'].mean():.2f}
• Maximum: {df['Distance'].max():.2f}
• Minimum: {df['Distance'].min():.2f}
• Std Dev: {df['Distance'].std():.2f}

Risk Assessment:
• Above Threshold: {(df['Distance'] > threshold).sum()} cases
• At/Below Threshold: {(df['Distance'] <= threshold).sum()} cases
    """
    
    ax6.text(0.1, 0.5, stats_text, transform=ax6.transAxes, fontsize=10,
             verticalalignment='center', bbox=dict(boxstyle="round,pad=0.3", 
             facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    # Save the plot
    output_file = 'flood_monitoring_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Visualization saved as: {output_file}")
    
    plt.show()
    
    return fig

def print_detailed_report(df, threshold):
    """Print a detailed analysis report."""
    print("\n" + "="*70)
    print("DETAILED FLOOD MONITORING ANALYSIS REPORT")
    print("="*70)
    
    print(f"Threshold Value: {threshold}")
    print(f"Total Measurements: {len(df)}")
    
    # Risk distribution
    high_risk_count = len(df[df['Distance'] > threshold])
    safe_count = len(df[df['Distance'] <= threshold])
    
    print(f"\nRisk Distribution:")
    print(f"  High Risk (Red):  {high_risk_count:4d} records ({high_risk_count/len(df)*100:5.1f}%)")
    print(f"  Safe (Green):     {safe_count:4d} records ({safe_count/len(df)*100:5.1f}%)")
    
    # Statistical analysis
    print(f"\nWater Level Statistics:")
    print(f"  Average:     {df['Distance'].mean():6.2f}")
    print(f"  Median:      {df['Distance'].median():6.2f}")
    print(f"  Maximum:     {df['Distance'].max():6.2f}")
    print(f"  Minimum:     {df['Distance'].min():6.2f}")
    print(f"  Std Dev:     {df['Distance'].std():6.2f}")
    print(f"  Range:       {df['Distance'].max() - df['Distance'].min():6.2f}")
    
    # Percentile analysis
    print(f"\nPercentile Analysis:")
    for p in [25, 50, 75, 90, 95, 99]:
        value = np.percentile(df['Distance'], p)
        status = "HIGH RISK" if value > threshold else "SAFE"
        print(f"  {p:2d}th percentile: {value:6.2f} ({status})")
    
    # Value distribution
    print(f"\nValue Distribution:")
    value_counts = df['Distance'].value_counts().sort_index()
    for value, count in value_counts.items():
        status = "HIGH RISK" if value > threshold else "SAFE"
        print(f"  {value:4.1f}: {count:4d} records ({count/len(df)*100:5.1f}%) - {status}")
    
    print("\n" + "="*70)

def save_labeled_data(df, output_file='flood_data_labeled.csv'):
    """Save the labeled data to a new CSV file."""
    # Select relevant columns for output
    output_df = df[['TimeStamp', 'LineCount', 'Distance', 'Status', 'Risk_Level']].copy()
    
    # Save to CSV
    output_df.to_csv(output_file, index=False)
    print(f"\nLabeled data saved to: {output_file}")
    
    # Show sample of labeled data
    print(f"\nSample of labeled data:")
    print(output_df.head(10).to_string(index=False))
    
    return output_df

def main():
    """Main function to run the complete flood monitoring analysis."""
    print("="*70)
    print("FLOOD MONITORING ANALYSIS WITH THRESHOLD-BASED RISK ASSESSMENT")
    print("="*70)
    
    try:
        # Load and prepare data
        df = load_and_prepare_data('Flood_Data.csv')
        
        # Set threshold (you can modify this value)
        THRESHOLD = 51.4
        
        # Analyze with threshold
        df_analyzed, threshold_used = analyze_threshold(df, THRESHOLD)
        
        # Create visualization
        fig = create_comprehensive_visualization(df_analyzed, threshold_used)
        
        # Print detailed report
        print_detailed_report(df_analyzed, threshold_used)
        
        # Save labeled data
        labeled_df = save_labeled_data(df_analyzed)
        
        print(f"\n{'='*70}")
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
        print(f"{'='*70}")
        print(f"• Visualization saved as: flood_monitoring_analysis.png")
        print(f"• Labeled data saved as: flood_data_labeled.csv")
        print(f"• Total records processed: {len(df_analyzed)}")
        print(f"• Threshold used: {threshold_used}")
        
    except FileNotFoundError:
        print("Error: Flood_Data.csv file not found!")
        print("Please ensure the CSV file is in the current directory.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()