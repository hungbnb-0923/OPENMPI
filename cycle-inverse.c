#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#define UPDATE_PIVOT 1
#define MAX_N 1000

const int TAG = 100;
const double _zeros[1000000] = {0};
int rank, size, N;
int winner[20];
int _winner = -1;

void _quit (const char *msg) {
	printf("%s\n", msg);
	exit(EXIT_FAILURE);
	MPI_Finalize();
}


void read_matrix_data(const char *path, double **_buff, int *N) {
	FILE *fp;
	fp = fopen(path, "r");
	fscanf(fp, "SHAPE = %d\n", N);
	const int n = *N;
	*_buff = (double*) malloc(n * 2*n * sizeof(double)); // matrix n row and 2*n col (origin matrix append identify matrix based-on Gausse Elimination)
	double *buff = *_buff;
	
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < 2*n; j++) {
			if (j < n) {
				if (j == n-1) fscanf(fp, "%lf\n", &buff[i * 2*n + j]);
				else fscanf(fp, "%lf ", &buff[i * 2*n + j]);
			} else {
				buff[i * 2*n + j] = 0;
				if (j % n == i) buff[i * 2*n + j] = 1;
			}
		}
	}

	fclose(fp);
}

void _toString(double *mtx, int R, int C, int fromR, int fromC) {
	for (int i = fromR; i < R; i++) {
		for (int j = fromC; j < C; j++) {
			printf("%f ", mtx[i * C + j]);
		}
		printf("\n");
	}
}

void toString(double *mtx, int R, int C) {
	_toString(mtx, R, C, 0, 0);
}

void swap_rows(double *matrix, int i, int j) {
	static double tmp[MAX_N];
	memcpy(tmp, &matrix[i * 2*N], 2*N * sizeof(double));
	memcpy(&matrix[i * 2*N], &matrix[j * 2*N], 2*N * sizeof(double));
	memcpy(&matrix[j * 2*N], tmp, 2*N * sizeof(double));
}

int main(int argc, char **argv) {
	double t_start, t_end, t_total;
	double comm_s, comm_total = 0;

	int code = MPI_Init(&argc, &argv);
	if (code != MPI_SUCCESS) _quit("Error");

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	double *matrix;
	if (rank == 0) {
		int test;
		if (argc < 2) _quit("Missing param: --test, EX: --test 5");
		else sscanf(argv[2], "%d", &test);

		char path2source[256]; 
		sprintf(path2source, "./data/test_%d", test);
		read_matrix_data(path2source, &matrix, &N);
	} 

	if (rank == 0) {
		t_start = MPI_Wtime();
	}

	comm_s = MPI_Wtime();
	MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
	comm_total += MPI_Wtime() - comm_s;

	// number of rows for each rank
	int num_rows = (int) N / size;
	double  *sub_matrix; 

	if (N % size && rank == 0) {
		sub_matrix = (double*) malloc((num_rows + N % size) * 2*N * sizeof(double));
		memcpy(sub_matrix, matrix, (N % size) * 2*N * sizeof(double));
	} else {
		sub_matrix = (double*) malloc(num_rows * 2*N * sizeof(double));
	}

	if (size == 1) memcpy(sub_matrix, matrix, N * 2*N * sizeof(double));
	else {
		comm_s = MPI_Wtime();
		for (int i = 0; i < num_rows; i++) {
			if (rank == 0) 
				MPI_Scatter(&matrix[(i + N % size) * 2*N * size], 2*N, MPI_DOUBLE, &sub_matrix[(i + N % size) * 2*N], 2*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
			else 
				MPI_Scatter(&matrix[(i + N % size) * 2*N * size], 2*N, MPI_DOUBLE, &sub_matrix[i * 2*N], 2*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
			
		}

		comm_total += MPI_Wtime() - comm_s;
	}

	// row var be used for elimination in each rank
	double *row = (double*) malloc(2*N * sizeof(double));

	// Gauss Elimination phase
	double pivot;
	int need_update_pivot;
	int resolvable_update_pivot;
	double scale;
	int local_row;
	int which_rank;

	double *inter_pivot_row;
	double *local_pivot_row;

	// Iterate over all rows
	for (int i = 0; i < N; i++) {
		// local row in sub-matrix be accessed
		local_row = i / size;

		// which rank does this row belong to ?
		which_rank = i % size;

		// Eliminate if the pivot belongs to this rank
		if (rank == which_rank) {
			// printf("Row: %d\n", i);
			_winner = which_rank;
			need_update_pivot = 0;
			resolvable_update_pivot = 0;

			pivot = sub_matrix[local_row * 2*N + i];
			local_pivot_row = &sub_matrix[local_row * 2*N];

			if (pivot == 0) {
				need_update_pivot = 1;

				// Intra update pivot
				for (int j = local_row + 1; j < num_rows; j++) {
					if (sub_matrix[j * 2*N + i]) {
						pivot = sub_matrix[j * 2*N + i];
						need_update_pivot = 0;
						resolvable_update_pivot = 1;
						swap_rows(sub_matrix, local_row, j);
						break;
					}
				}
			}

			comm_s = MPI_Wtime();
			MPI_Bcast(&need_update_pivot, 1, MPI_INT, which_rank, MPI_COMM_WORLD);
			comm_total += MPI_Wtime() - comm_s;

			if (need_update_pivot) {
				// Inter update pivot 
				comm_s = MPI_Wtime();
				MPI_Gather(&resolvable_update_pivot, 1, MPI_INT, winner, 1, MPI_INT, which_rank, MPI_COMM_WORLD);
				comm_total += MPI_Wtime() - comm_s;

				for (int i = 0; i < size; i++) {
					if (winner[i] == 1) {
						_winner = i;
						break;
					}
				}

				// bcast the choosen winner is the first winner
				comm_s = MPI_Wtime();
				MPI_Bcast(&_winner, 1, MPI_INT, which_rank, MPI_COMM_WORLD);
				comm_total += MPI_Wtime() - comm_s;

				if (_winner == -1) {
					break;
				}

				// send local pivot to the choosen winner
				comm_s = MPI_Wtime();
				MPI_Send(local_pivot_row, 2*N, MPI_DOUBLE, _winner, UPDATE_PIVOT, MPI_COMM_WORLD);
				comm_total += MPI_Wtime() - comm_s;

				// recv inter pivot from the choosen winner
				inter_pivot_row = (double*) malloc(2*N * sizeof(double));

				comm_s = MPI_Wtime();
				MPI_Recv(inter_pivot_row, 2*N, MPI_DOUBLE, _winner, UPDATE_PIVOT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				comm_total += MPI_Wtime() - comm_s;

				// update local pivot 
				memcpy(local_pivot_row, inter_pivot_row, 2*N * sizeof(double));

				pivot = sub_matrix[local_row * 2*N + i];
			}

			// Devide the rest of the row by the pivot
			for (int j = i + 1; j < 2*N; j++) {
				sub_matrix[local_row * 2*N + j] /= pivot;
			}

			sub_matrix[local_row * 2*N + i] = 1;
			memcpy(row, &sub_matrix[local_row * 2*N], 2*N * sizeof(double));

			// Broadcast this row to all the ranks
			comm_s = MPI_Wtime();
			MPI_Bcast(row, 2*N, MPI_DOUBLE, which_rank, MPI_COMM_WORLD);
			comm_total += MPI_Wtime() - comm_s;

			// Eliminate for the other rows mapped to this rank
			for (int j = local_row + 1; j < num_rows; j++) {
				scale = sub_matrix[j * 2*N  + i];
				if (scale == 0) continue;

				for (int k = i + 1; k < 2*N; k++) {
					sub_matrix[j * 2*N + k] -= scale * row[k];
				}
				sub_matrix[j * 2*N + i] = 0;
			}
		} else {
			// ? update pivot 
			comm_s = MPI_Wtime();
			MPI_Bcast(&need_update_pivot, 1, MPI_INT, which_rank, MPI_COMM_WORLD);
			comm_total += MPI_Wtime() - comm_s;
			if (need_update_pivot) {
				// Check wether resolvalbe update pivot or not
				resolvable_update_pivot = 0;
				for (int j = local_row; j < num_rows; j++) {
					if ((which_rank < rank) || (j > local_row)) {
						if (sub_matrix[j * 2*N + i]) {
							local_pivot_row = &sub_matrix[j * 2*N + i];
							resolvable_update_pivot = 1;
							break;
						}
					}
				}

				comm_s = MPI_Wtime();
				MPI_Gather(&resolvable_update_pivot, 1, MPI_INT, winner, 1, MPI_INT, which_rank, MPI_COMM_WORLD);
				comm_total += MPI_Wtime() - comm_s;

				comm_s = MPI_Wtime();
				MPI_Bcast(&_winner, 1, MPI_INT, which_rank, MPI_COMM_WORLD);
				comm_total += MPI_Wtime() - comm_s;

				if (_winner == -1) {
					break;
				}

				if (_winner == rank) {
					// recv inter pivot from the which_rank
					inter_pivot_row = (double*) malloc(2*N * sizeof(double));

					comm_s = MPI_Wtime();
					MPI_Recv(inter_pivot_row, 2*N, MPI_DOUBLE, which_rank, UPDATE_PIVOT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					comm_total += MPI_Wtime() - comm_s;

					// send local pivot to the which_rank
					comm_s = MPI_Wtime();
					MPI_Send(local_pivot_row, 2*N, MPI_DOUBLE, which_rank, UPDATE_PIVOT, MPI_COMM_WORLD);
					comm_total += MPI_Wtime() - comm_s;

					// update local pivot 
					memcpy(local_pivot_row, inter_pivot_row, 2*N * sizeof(double));
				}
			}

			// Recevice a row to use for elimination
			comm_s = MPI_Wtime();
			MPI_Bcast(row, 2*N, MPI_DOUBLE, which_rank, MPI_COMM_WORLD);
			comm_total += MPI_Wtime() - comm_s;

			// Eliminate for all the rows mapped to this rank
			for (int j = local_row; j < num_rows; j++) {
				if ((which_rank < rank) || (j > local_row)) {
					scale = sub_matrix[j * 2*N + i];

					for (int k = i + 1; k < 2*N; k++) {
						sub_matrix[j * 2*N + k] -= scale * row[k];
					}

					sub_matrix[j * 2*N + i] = 0;
				}
			}
		}
	}

	MPI_Barrier(MPI_COMM_WORLD);

	if (_winner != -1) {
		// Back elimination phase
		for (int i = N-1; i >= 0; i--) {
			local_row = i / size;
			which_rank = i % size;

			if (rank == which_rank) {
				memcpy(row, &sub_matrix[local_row * 2*N], 2*N * sizeof(double));

				// Broadcast this row to all the ranks
				comm_s = MPI_Wtime();
				MPI_Bcast(row, 2*N, MPI_DOUBLE, which_rank, MPI_COMM_WORLD);
				comm_total += MPI_Wtime() - comm_s;

				for (int j = local_row - 1; j >= 0; j--) {
					scale = sub_matrix[j * 2*N + i];

					for (int k = N; k < 2*N; k++) {
						sub_matrix[j * 2*N + k] -= scale * row[k];
					}

					sub_matrix[j * 2*N + i] = 0;
				}

			} else {
				comm_s = MPI_Wtime();
				MPI_Bcast(row, 2*N, MPI_DOUBLE, which_rank, MPI_COMM_WORLD);
				comm_total += MPI_Wtime() - comm_s;

				for (int j = local_row; j >= 0; j--) {
					if ((which_rank > rank) || (j < local_row)) {
						scale = sub_matrix[j * 2*N + i];

						for (int k = N; k < 2*N; k++) {
							sub_matrix[j * 2*N + k] -= scale * row[k];
						}

						sub_matrix[j * 2*N + i] = 0;
					}
				}
			}
		}

		MPI_Barrier(MPI_COMM_WORLD);

		if (size == 1) {
			memcpy(matrix, sub_matrix, N * 2*N * sizeof(double));
		} else {
			// Gather "size" row at a time
			if (rank == 0) {
				memcpy(matrix, sub_matrix, (N % size) * 2*N * sizeof(double));
				sub_matrix = &sub_matrix[(N % size) * 2*N];
			}

			comm_s = MPI_Wtime();
			for (int i = 0; i < num_rows; i++) {
				MPI_Gather(&sub_matrix[i * 2*N], 2*N, MPI_DOUBLE, &matrix[(i + N % size) * 2*N * size], 2*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
			}
			comm_total += MPI_Wtime() - comm_s;
		}

		if (rank == 0) {
			t_end = MPI_Wtime();
			t_total = t_end - t_start;
		}
	} 

	if (rank == 0) {
		if (_winner == -1) {
			printf("\nNOT EXIST\n");
		} else {
			// toString(matrix, N, 2*N);

			/*
			printf("Total time:  %f s\n", t_total);
			printf("Communication time:  %f s\n", comm_total);
			*/
			printf("%d, %f, %f\n", size, t_total, comm_total);
		}

		free(matrix);
	}

	free(sub_matrix);
	free(row);


	MPI_Finalize();

	return 0;
}
