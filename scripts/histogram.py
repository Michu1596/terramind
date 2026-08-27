import os
import pandas as pd
import matplotlib.pyplot as plt

def main():
    input_path = "results/H_results.csv"
    output_dir = "scripts/results"
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the data
    df = pd.read_csv(input_path)
    
    # Filter for iteration == 12
    df_iter12 = df[df["iteration"] == 12].copy()
    
    # 1. Create Histogram of moran_I for iteration = 12
    plt.figure(figsize=(8, 5))
    plt.hist(df_iter12["moran_I"], bins=20, color="#2b5c8f", edgecolor="black", alpha=0.8)
    plt.title("Histogram of Moran's I for Altitude on depth 12", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Moran's I", fontsize=12)
    plt.ylabel("Occurrences", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    
    # Save output as vector PDF and PNG
    pdf_path = os.path.join(output_dir, "moran_I_iter12_histogram.pdf")
    png_path = os.path.join(output_dir, "moran_I_iter12_histogram.png")
    
    plt.savefig(pdf_path, format="pdf", dpi=300)
    plt.savefig(png_path, format="png", dpi=300)
    plt.close()
    
    print(f"Saved histogram to {pdf_path} and {png_path}")
    
    # 2. Save filtered table in LaTeX format
    latex_path = os.path.join(output_dir, "moran_I_iter12_table.tex")
    
    latex_str = df_iter12.to_latex(
        index=False,
        caption="Moran's I Results for Iteration 12 Across Seeds",
        label="tab:moran_iter12",
        column_format="lccccc",
        float_format="%.4f"
    )
    
    with open(latex_path, "w") as f:
        f.write(latex_str)
        
    print(f"Saved LaTeX table to {latex_path}")

if __name__ == "__main__":
    main()