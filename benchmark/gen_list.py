def generate_orthogonal_test_list(output_file):
    """
    Generate an orthogonal test case list file with parameters in the specified order.
    """
    # Define parameter ranges
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    seqlen_ks = [200, 400, 1024, 1680, 2560, 3560, 4096, 4960, 5500, 6120, 6600, 7120, 8192] # [4096, 6144, 8192] 
    num_heads = [8, 16, 32, 64, 128] # [32, 48, 64]

    # Fixed parameters
    seqlen_q = 1
    num_heads_kv = 1
    head_dim = 576
    head_dim_v = 512
    causal = 1
    dtype = "fp16"

    with open(output_file, 'w') as f:
        # Iterate through all orthogonal combinations
        for bs in batch_sizes:
            for sk in seqlen_ks:
                for nh in num_heads:
                    # Build the parameter string in the required order
                    params_str = (
                        f"batch_size:{bs},"
                        f"seqlen_q:{seqlen_q},"
                        f"seqlen_k:{sk},"
                        f"num_heads:{nh},"
                        f"num_heads_kv:{num_heads_kv},"
                        f"head_dim:{head_dim},"
                        f"head_dim_v:{head_dim_v},"
                        f"causal:{causal},"
                        f"dtype:{dtype},block_size:64"
                    )
                    line = f"[MLA] --format={params_str}\n"
                    f.write(line)

    print(f"Successfully generated {output_file}")


if __name__ == "__main__":
    # Specify output file name
    output_filename = "test2.list"

    # Call the function to generate the orthogonal test case file
    generate_orthogonal_test_list(output_filename)
