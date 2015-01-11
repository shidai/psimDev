// functions used to simulate dynamic spectrum  
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <fftw3.h>
//#include <gsl/gsl_rng.h>
//#include <gsl/gsl_randist.h>
#include "dynSpecSim.h"
#include "T2toolkit.h"
//#include "psim.h"

int calculateScintScale (acfStruct *acfStructure, controlStruct *control)
{
	printf ("Starting simulating dynamic spectrum\n");
	acfStructure->bw = fabs(control->obsBW); // MHz
	acfStructure->f0 = control->scint_freqbw;  // MHz
	acfStructure->tint = control->nsub*control->tsubRequested;  // s
	acfStructure->t0 = control->scint_ts; // MHz
	printf ("Scintillation bandwidth: %lf (MHz)\n", acfStructure->f0);
	printf ("Scintillation time-scale: %lf (s)\n", acfStructure->t0);

	acfStructure->nchn = control->nchan;
	acfStructure->nsubint = control->nsub;

	preAllocateMemory (acfStructure);
	allocateMemory (acfStructure);

	calACF (acfStructure);
	power (acfStructure);
	simDynSpec (acfStructure);

	return 0;
}

int calACF (acfStruct *acfStructure)
{
	printf ("Calculating ACF\n");
	int i,j;
	int ns = acfStructure->ns;
	int nf = acfStructure->nf;
	double steps = acfStructure->steps;
	double stepf = acfStructure->stepf;
	double *acf;
	acf = (double *)malloc(sizeof(double)*ns*nf);

	for (i = 0; i < ns; i++)
	{
		acfStructure->s[i] = -acfStructure->size[1]+i*steps;
	}

  for (i = 0; i < nf; i++)
	{
		acfStructure->f[i] = -acfStructure->size[0]+i*stepf;
	}

	double rand;
	rand = acfStructure->phaseGradient;
	//printf ("%lf\n",rand);
	
	int n = 0;
	for (i = 0; i < nf; i++)
	{
		for (j = 0; j < ns; j++)
		{
			//acf[n] = exp(-pow((pow(acfStructure->s[j],2.5)+pow(acfStructure->f[i],1.5)),2.0/3.0));
			//acf[n] = exp(-pow((pow(fabs(acfStructure->s[j]),2.5)+pow(fabs(acfStructure->f[i]),1.5)),2.0/3.0));
			acf[n] = exp(-pow((pow(fabs(acfStructure->s[j]+2.0*rand*0.4*acfStructure->f[i]),2.5)+pow(fabs(acfStructure->f[i]),1.5)),2.0/3.0));
			n++;
		}
	}

	//////////////////////////////////////////////////////////////////

	for (i = 0; i < nf; i++)
	{
		for (j = 0; j < ns; j++)
		{
			if (i >= (int)ceil(nf/2.0) && j >= (int)ceil(ns/2.0))
			{
				acfStructure->acf2d[i*ns+j] = acf[ns*(i-(int)ceil(nf/2.0))+(j-(int)ceil(ns/2.0))];
			}
			else if  (i >= (int)ceil(nf/2.0) && j < (int)ceil(ns/2.0))
			{
				acfStructure->acf2d[i*ns+j] = acf[ns*(i-(int)ceil(nf/2.0))+(j+(int)floor(ns/2.0))];
			}
			else if  (i < (int)ceil(nf/2.0) && j < (int)ceil(ns/2.0))
			{
				acfStructure->acf2d[i*ns+j] = acf[ns*(i+(int)floor(nf/2.0))+(j+(int)floor(ns/2.0))];
			}
			else if  (i < (int)ceil(nf/2.0) && j >= (int)ceil(ns/2.0))
			{
				acfStructure->acf2d[i*ns+j] = acf[ns*(i+(int)floor(nf/2.0))+(j-(int)ceil(ns/2.0))];
			}
		}
	}

	free(acf);

	return 0;
}


int dft2d (acfStruct *acfStructure, fftw_complex *out)
{
	int i;
	int n0 = acfStructure->nf;
	int n1 = acfStructure->ns;
	double *in;
	in = (double *)malloc(sizeof(double)*n0*n1);

	fftw_plan p;
	
	p = fftw_plan_dft_r2c_2d (n0, n1, in, out, FFTW_ESTIMATE);
	//p = fftw_plan_dft_r2c_2d (n0, n1, in, out, FFTW_MEASURE);

	for (i = 0; i < n0*n1; i++)
	{
		in[i] = acfStructure->acf2d[i];
	}

	fftw_execute(p);

	fftw_destroy_plan(p);
	free (in);
  
	return 0;
}

int idft2d (acfStruct *acfStructure)
{
	int i;
	int n0 = acfStructure->nf;
	int n1 = acfStructure->ns;

	fftw_complex *in;
	in = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*n0*n1);

	fftw_plan p;
	
	p = fftw_plan_dft_2d (n0, n1, in, acfStructure->intensity, FFTW_BACKWARD, FFTW_ESTIMATE);
	//p = fftw_plan_dft_2d (n0, n1, in, acfStructure->intensity, FFTW_BACKWARD, FFTW_MEASURE);

	for (i = 0; i < n0*n1; i++)
	{
		in[i][0] = acfStructure->eField[i][0];
		in[i][1] = acfStructure->eField[i][1];
	}

	fftw_execute(p);

	fftw_destroy_plan(p);
	fftw_free (in);
  
	return 0;
}

int power (acfStruct *acfStructure)
{
	int i;
	int nf = acfStructure->nf;
	int ns = acfStructure->ns;
	/////////////////////////////////////////////////////////////////////////////////

  fftw_complex *out;
	
	out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*nf*((int)(ns/2)+1));
	
	dft2d (acfStructure, out);

	int n, j;
	n = 0;
	for (i = 0; i < nf; i++)
	{
		for (j = 0; j < ns; j++)
		{
			if (j < (int)(ns/2)+1)
			{
				acfStructure->psrt[n] = sqrt(sqrt(pow(out[i*((int)(ns/2)+1)+j][0],2.0)+pow(out[i*((int)(ns/2)+1)+j][1],2.0)));
			}
			else
			{
				if (i == 0)
				{
					acfStructure->psrt[n] = sqrt(sqrt(pow(out[i*((int)(ns/2)+1)+ns-j][0],2.0)+pow(out[i*((int)(ns/2)+1)+ns-j][1],2.0)));
				}
				else
				{
					acfStructure->psrt[n] = sqrt(sqrt(pow(out[(nf-i)*((int)(ns/2)+1)+ns-j][0],2.0)+pow(out[(nf-i)*((int)(ns/2)+1)+ns-j][1],2.0)));
				}
			}
			//printf ("%lf ", acfStructure->psrt[n]);
			n++;
		}
		//printf ("\n");
	}

	fftw_free(out); 

	return 0;
}

void allocateMemory (acfStruct *acfStructure)
{
	int ns, nf;
	int nchn, nsubint;

	ns = acfStructure->ns;
	nf = acfStructure->nf;
	nchn = acfStructure->nchn;
	nsubint = acfStructure->nsubint;
	
	acfStructure->s = (double *)malloc(sizeof(double)*ns);
	acfStructure->f = (double *)malloc(sizeof(double)*nf);
	acfStructure->acf2d = (double *)malloc(sizeof(double)*ns*nf);
	acfStructure->psrt = (double *)malloc(sizeof(double)*ns*nf);
	acfStructure->eField = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*nf*ns);
	acfStructure->intensity = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*nf*ns);
	
	acfStructure->dynSpec = (double **)fftw_malloc(sizeof(double *)*nf);
	acfStructure->dynSpecWindow = (double **)fftw_malloc(sizeof(double *)*nchn);

	int i;
	for (i = 0; i < nf; i++)
	{
		acfStructure->dynSpec[i] = (double *)fftw_malloc(sizeof(double)*ns);
	}

	for (i = 0; i < nchn; i++)
	{
		acfStructure->dynSpecWindow[i] = (double *)fftw_malloc(sizeof(double)*nsubint);
	}
}

void deallocateMemory (acfStruct *acfStructure)
{
	int nf = acfStructure->nf;
	int nchn = acfStructure->nchn;
	
	fftw_free(acfStructure->s); 
	fftw_free(acfStructure->f); 
	fftw_free(acfStructure->eField); 
	fftw_free(acfStructure->intensity); 
	free(acfStructure->acf2d);
	free(acfStructure->psrt);

	int i;
	for (i = 0; i < nf; i++)
	{
		free(acfStructure->dynSpec[i]);
	}

	for (i = 0; i < nchn; i++)
	{
		free(acfStructure->dynSpecWindow[i]);
	}

	free(acfStructure->dynSpec);
}

int simDynSpec (acfStruct *acfStructure)
{
	printf ("Simulating dynamic spectrum\n");
	int nf = acfStructure->nf;
	int ns = acfStructure->ns;
	int nchn = acfStructure->nchn;
	int nsubint = acfStructure->nsubint;

	long seed;
	
	int i;
	int j;
	seed = TKsetSeed();
	//printf ("seed %ld\n",seed);

	for (i = 0; i < nf*ns; i++)
	{
		//acfStructure->eField[i][0] = acfStructure->psrt[i];
		//acfStructure->eField[i][1] = acfStructure->psrt[i];
		acfStructure->eField[i][0] = acfStructure->psrt[i]*TKgaussDev(&seed);
		acfStructure->eField[i][1] = acfStructure->psrt[i]*TKgaussDev(&seed);
		//printf ("%lf\n",TKgaussDev(&seed));
	}

	/////////////////////////////////////////////////////////////////////////////////

	// ifft
	idft2d (acfStructure);

	// form the matrix and normalize
	int n = 0;
	double sum = 0.0;

	for (i = 0; i < nf; i++)
	{
		for (j = 0; j < ns; j++)
		{
			acfStructure->dynSpec[i][j] = pow(acfStructure->intensity[n][1]/(nf*ns),2.0)+pow(acfStructure->intensity[n][0]/(nf*ns),2.0);
			//fprintf (fp, "%lf  ", acfStructure->dynSpec[i][j]);
			sum += pow(acfStructure->intensity[n][1]/(nf*ns),2.0)+pow(acfStructure->intensity[n][0]/(nf*ns),2.0);
			n++;
		}
		//fprintf (fp, "\n");
	}

	sum = sum/n;
	printf ("Normalization %.10lf\n",sum);

	// choose a subwindow
	double rand, rand2;
	//printf ("seed %ld\n",seed);
	rand = TKgaussDev(&seed);
	rand2 = rand - floor(rand);

	int nf0 = (int)(rand2*(nf-nchn));
	int ns0 = (int)(rand2*(ns-nsubint));
	
	for (i = 0; i < nchn; i++)
	{
		for (j = 0; j < nsubint; j++)
		{
			acfStructure->dynSpecWindow[i][j] = acfStructure->dynSpec[i+nf0][j+ns0]/sum;
			//fprintf (fp, "%.10lf  ", acfStructure->dynSpec[i][j]/sum);
		}
	}

	return 0;
}

int windowSize (acfStruct *acfStructure, double *size)
{
	double bw, f0, tint, t0;
	bw = acfStructure->bw;
	f0 = acfStructure->f0;
	tint = acfStructure->tint;
	t0 = acfStructure->t0;

	if ( (bw/f0) > 6 )
	{
		size[0] = bw/f0;
	}
	else
	{
		size[0] = 6.0;
	}
			  
	if ( (tint/t0) > 6 )
	{
		size[1] = tint/t0;
	}
	else
	{
		size[1] = 6.0;
	}

	double ratio[2];
	calSize (acfStructure, size, ratio);
	//printf ("f0 ratio: %lf\n", ratio[0]);
	//printf ("s0 ratio: %lf\n", ratio[1]);

	while (ratio[0] >= 1e-6 || ratio[1] >= 1e-6)
	{
		size[0] = 1.05*size[0];
		size[1] = 1.05*size[1];
		calSize (acfStructure, size, ratio);
		//printf ("f0 ratio: %lf\n", ratio[0]);
		//printf ("s0 ratio: %lf\n", ratio[1]);
	}

	acfStructure->size[0] = size[0];
	acfStructure->size[1] = size[1];

	return 0;
}

int calSize (acfStruct *acfStructure, double *size, double *ratio)
{
	int i;
	double rand = acfStructure->phaseGradient;
	double steps = acfStructure->steps;
	double stepf = acfStructure->stepf;

	int nf = (int)(size[0]*2/stepf)+1;
	int ns = (int)(size[1]*2/steps)+1;

	double s[ns], acfs[ns], smax;
	double f[nf], acff[nf], fmax;
	//double c; // value at the center

	for (i = 0; i < ns; i++)
	{
		s[i] = -size[1]+i*steps;
	}

  for (i = 0; i < nf; i++)
	{
		f[i] = -size[0]+i*stepf;
	}

	//c = exp(-pow((pow(fabs(s[(int)(ns/2)]+2.0*rand*0.4*f[(int)(nf/2)]),2.5)+pow(fabs(f[(int)(nf/2)]),1.5)),2.0/3.0));

	for (i = 0; i < nf; i++)
	{
		acff[i] = exp(-pow((pow(fabs(s[0]+2.0*rand*0.4*f[i]),2.5)+pow(fabs(f[i]),1.5)),2.0/3.0));
	}

	for (i = 0; i < ns; i++)
	{
		acfs[i] = exp(-pow((pow(fabs(s[i]+2.0*rand*0.4*f[0]),2.5)+pow(fabs(f[0]),1.5)),2.0/3.0));
	}

	smax = find_peak_value (ns, acfs);
	fmax = find_peak_value (nf, acff);

	ratio[0] = fmax;
	ratio[1] = smax;

	return 0;
}

double find_peak_value (int n, double *s)
{
	int i;
	double temp[n];

	for (i = 0; i < n; i++)
	{
		temp[i] = s[i];
	}

	
	double a, b, c;
	for (i = 0; i < n-1; i++)
	{
		a = temp[i];
		b = temp[i+1];
															
		c = (a >= b ? a : b);
		
		temp[i+1] = c;
	
	}
	
	return temp[n-1];
}
															
void preAllocateMemory (acfStruct *acfStructure)
{
	long seed;
	//seed = -1420603014;
	seed = TKsetSeed();
	printf ("Seed %ld\n",seed);
	acfStructure->phaseGradient = TKgaussDev(&seed);
	//acfStructure->phaseGradient = 0.0;
	printf ("Phase gradient: %lf\n", acfStructure->phaseGradient);

	double bw, f0, tint, t0;
	int nchn, nsubint;

	bw = acfStructure->bw;
	f0 = acfStructure->f0;
	tint = acfStructure->tint;
	t0 = acfStructure->t0;

	nchn = acfStructure->nchn;
	nsubint = acfStructure->nsubint;

	double steps = (tint/t0)/nsubint;
	double stepf = (bw/f0)/nchn;
	acfStructure->steps = steps;
	acfStructure->stepf = stepf;
	//printf ("%lf\n", steps);
	//printf ("%lf\n", stepf);
	
	double size[2]; // sampling boundary
	windowSize (acfStructure, size);
	printf ("f0 size: %lf\n", size[0]);
	printf ("s0 size: %lf\n", size[1]);

	int nf = (int)(size[0]*2/stepf)+1;
	int ns = (int)(size[1]*2/steps)+1;
	//printf ("%d\n", ns);
	//printf ("%d\n", nf);

	acfStructure->ns = ns;
	acfStructure->nf = nf;
}

