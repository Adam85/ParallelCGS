void getColumn_1d( double * A,  double *a, int rows, int colid);
void setColumn_1d( double * A,  double *a, int rows, int colid, int zid, int cols);
void updatev_1d( double * A, int rows, int colid, int zid, int cols);

void  cgsro_sequential( double * A_1d,  double * Q_1d, int steps, int m, int n, double ** timer);

