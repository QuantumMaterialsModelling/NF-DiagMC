#pragma once

struct Histogram{
public:
	int dim;
	int size;	

	Histogram(int size, double beg, double end, int dim = 1);


	void add(double x, int which = 0);
	void add(int idx, int which = 0);
	void print(const char *path);

private:
	int search(double x);

private:
	double dx_;
	double *x_;
	int **hist_;

};
