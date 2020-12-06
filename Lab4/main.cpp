#include <iostream>
#include <string>
#include <random>
#include <numeric>
#include <functional>
#include <execution>
#include <cmath>
#include <mpi.h>

#define MY_MPI_REAL MPI_DOUBLE
using real = double;

using namespace std;
constexpr real MIN_ELEM = -1.0, MAX_ELEM = 1.0, FACTOR = 0.0000001;
constexpr int TAYLOR_TERMS = 80;

// Using Taylor expansion of the cosine function
real myCos(real x)
{
	real res = 1., cur = 1., mxSq = -x * x;
	for (int i = 1; i < TAYLOR_TERMS; ++i)
		res += (cur *= (mxSq / (2.*i) / (2.*i - 1.)));
	return res;
}

int main(int argc, char* argv[])
{
	if (argc != 2)
	{
		cerr << "You must give 1 argument - size of array" << endl;
		return -1;
	}
	const int n = atoi(argv[1]);
	MPI_Init(&argc, &argv);

	int procCnt, curProcRank, curElems;
	MPI_Comm_size(MPI_COMM_WORLD, &procCnt);
	MPI_Comm_rank(MPI_COMM_WORLD, &curProcRank);
	curElems = n / procCnt;
	if (curProcRank < n % procCnt)
		++curElems;
	const string procStr = "Process " + std::to_string(curProcRank) + ": ";
	if (curProcRank == 0)
	{
		cout << "Number of processes: " << procCnt << endl;
		cout << "Number of matrix data elements N: " << n << endl;
	}
	cout << procStr << "processing " << curElems << " elements" << endl;
	vector<real> a(1), x(1), ap(curElems), xp(curElems);

	double startTime, endTime;
	if (curProcRank == 0)
	{
		a.resize(n);
		x.resize(n);

		random_device rd;
		mt19937 engine(rd());
		uniform_real_distribution<real> dist(MIN_ELEM, MAX_ELEM);
		auto genFn = bind(ref(dist), ref(engine));
		generate(begin(a), end(a), genFn);
		
		for (int i = 0; i < n; ++i)
			x[i] = FACTOR * i;

		startTime = MPI_Wtime();
	}

	// Term function using our myCos function for cosine
	auto termFn = [](real a, real x) {return a * myCos(x); };

	MPI_Scatter(&a[0], curElems, MY_MPI_REAL, &ap[0], curElems, MY_MPI_REAL,
		0, MPI_COMM_WORLD);
	MPI_Scatter(&x[0], curElems, MY_MPI_REAL, &xp[0], curElems, MY_MPI_REAL,
		0, MPI_COMM_WORLD);
	const real sum = transform_reduce(begin(ap), end(ap), begin(xp), 0.0,
		plus(), termFn);
	real total = 0.0;
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Reduce(&sum, &total, 1, MY_MPI_REAL, MPI_SUM, 0, MPI_COMM_WORLD);

	if (curProcRank == 0)
	{
		endTime = MPI_Wtime();
		cout << endTime - startTime << "s elapsed" << endl;
		cout << "Overall sum of series using MPI with myCos: " << total << endl;

		startTime = MPI_Wtime();
		const real resSeq = transform_reduce(
			begin(a), end(a), begin(x), 0.0, plus(), termFn);
		endTime = MPI_Wtime();
		cout << "Sum of series using non-parallel std::transform_reduce with myCos: "
			<< resSeq << " in " << endTime - startTime << "s " << endl;
		
		startTime = MPI_Wtime();
		const real resPar = transform_reduce(execution::par,
			begin(a), end(a), begin(x), 0.0, plus(), termFn);
		endTime = MPI_Wtime();
		cout << "Sum of series using parallel std::transform_reduce with myCos: "
			<< resPar << " in " << endTime - startTime << "s " << endl;

		// Term function using standard cosine function
		auto termFnSt = [](real a, real x) {return a * cos(x); };

		startTime = MPI_Wtime();
		const real resSeqSt = transform_reduce(
			begin(a), end(a), begin(x), 0.0, plus(), termFnSt);
		endTime = MPI_Wtime();
		cout << "Sum of series using non-parallel std::transform_reduce with cos: "
			<< resSeqSt << " in " << endTime - startTime << "s " << endl;

		startTime = MPI_Wtime();
		const real resParSt = transform_reduce(execution::par,
			begin(a), end(a), begin(x), 0.0, plus(), termFnSt);
		endTime = MPI_Wtime();
		cout << "Sum of series using parallel std::transform_reduce with cos: "
			<< resParSt << " in " << endTime - startTime << "s " << endl;
	}

	MPI_Finalize();
	return 0;
}