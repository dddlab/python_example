#include <iostream>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <chrono>
#include <omp.h>

int main() {
    int p = 5000;
    int num_threads = omp_get_max_threads(); // Set the number of threads
    omp_set_num_threads(num_threads);

    // Define a sparse matrix
    Eigen::SparseMatrix<double> sparseMatrix(p, p);
    sparseMatrix.insert(0, 0) = 1;
    sparseMatrix.insert(1, 1) = 2;

    // Define a dense matrix
    Eigen::MatrixXd denseMatrix = Eigen::MatrixXd::Random(p, p);

    // Start measuring total time
    auto total_start = std::chrono::steady_clock::now();

    #pragma omp parallel
    {   
        for (int i = 0; i < 5; ++i) {
            // Start measuring time for each iteration
            auto start = std::chrono::steady_clock::now();

            // Perform the multiplication
            Eigen::MatrixXd result = sparseMatrix * denseMatrix;

            // End measuring time for each iteration
            auto end = std::chrono::steady_clock::now();

            // Calculate the elapsed time for each iteration
            std::chrono::duration<double> elapsed_seconds = end - start;
            std::cout << "Thread " << omp_get_thread_num() << " - Iteration " << i + 1 << " - Elapsed time: " << elapsed_seconds.count() << "s\n";
        }
    }

    // End measuring total time
    auto total_end = std::chrono::steady_clock::now();

    // Calculate total wall clock time
    std::chrono::duration<double> total_elapsed_seconds = total_end - total_start;

    std::cout << "Total time: " << total_elapsed_seconds.count() << "s\n"; // Print total elapsed time
    
    return 0;
}