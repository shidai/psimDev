#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "fitsio.h"
#include "tempo2pred.h"
#include <fftw3.h>

typedef struct vMises {
  double concentration; // Concentration of each component
  double height;        // Height of each component
  double centroid;      // Centroid of each component
  double concentration_err; // error of Concentration 
  double height_err;        // error of Height
  double centroid_err;      // error of Centroid
} vMises;

typedef struct component {
  int Comp;            
  int nVm;             // Number of components for each channel for each Stokes
  vMises *vonMises;
  int vmMemoryAllocated;
  int nVmAllocated;
} component;

typedef struct polStruct {
  int stokes;            // 1 = I, 2 = Q, 3 = U, 4 = V
  int nComp;             // Number of components for each channel for each Stokes
  int allVm;
  component *comp;
  int compMemoryAllocated;
  int nCompAllocated;
} polStruct;

typedef struct channelStruct {
  int nstokes;             // 1 = I, 4 = I,Q,U,V
  double freqLow;
  double freqHigh;
  polStruct *pol; 
  int polMemoryAllocated;
  int nPolAllocated;
} channelStruct;

typedef struct tmplStruct {
  // Common to all templates
  char dte[1024];      // Date template made
  char user[1024];   // Person who made the template
  float templateVersion; // Version of template header
  char source[128]; // Source name
  char profileFile[1024]; // Profile file name
  char units[1024];   // Unit definition
  double dedispersed; // = 0 by default

  int nchan; // Number of channels
  channelStruct *channel; // Channels
  int channelMemoryAllocated;
  int nChannelAllocated;
} tmplStruct;


typedef struct polarisation {
  double *val;
  int nPhaseBins;
} polarisation;


typedef struct channel {
  polarisation *pol; 
  int npol;
} channel;

typedef struct controlStruct {
  char  template[1024];
  char  primaryHeaderParams[1024]; // Filename of file containing primary header parameters
  char  exact_ephemeris[1204];
  char  fname[1024];               // Output filename for observation
  char  src[1024];                 // Source name
  char  type[1024];                // PSR or CAL
  int   nbin;
  int   nchan;
  int   npol;
  int   nsub;
  double cFreq;
  double obsBW;
  double segLength;
  int nfreqcoeff;
  int ntimecoeff;
  int stt_imjd;
  double stt_smjd;
  double stt_offs;
  double tsubRequested;  // Time for each subintegration requested (s)
  double tsub; // Time for each subintegration [modified to give an integral number of pulses per subint] (s)
  double dm;
  double whiteLevel;
	
	double tsys;   // system temperature
	double tsky;   // sky temperature
	double gain;   
	double radioNoise;

	double cFlux; // flux density at 1.4GHz, mJy
	double si;  // spectral index
	double *flux;  // flux density in each channel

  double scint_freqbw;
  double scint_ts;

	int bat;  // default: bat = 0, simulate at the observatory
  // Automatically calculated
  long double period;
  long double *phaseOffset;
} controlStruct;

typedef struct acfStruct {
	double phaseGradient;
	double bw; // observing bandwidth
	double f0; // scintillation bandwidth
	double tint; // integration time
	double t0; // scintillation time-scale
	int nchn;
	int nsubint;
	int ns; // sampling number of spatial scale 
	int nf; // sampling number of frequency scale
	double size[2]; // sampling boundary
	double steps;
	double stepf;
	double *s; // spatial scale
	double *f; // bw scale
	double *acf2d;  // ACF 
	double *psrt;  // power spactrum
	fftw_complex *eField; // complex electric field
	fftw_complex *intensity;  // intensity 
	double **dynSpec; // dynamic spectrum 
	double **dynSpecWindow; // dynamic spectrum window, nchn*nsubint dimension
} acfStruct;

void initialiseTemplate(tmplStruct *tmpl);
void readTemplate(char *file,tmplStruct *tmpl);
double evaluateTemplateComponent(tmplStruct *tmpl,double phi,int chan,int stokes,int comp,double phiRot);
double evaluateTemplate(controlStruct *control, tmplStruct *tmpl, int chan, int pol, int bin);
//double evaluateTemplate(controlStruct *control,int chan,int pol,int bin);

void createFitsFile(fitsfile *fptr,controlStruct *control);
void writeChannels(channel *chan,controlStruct *control,fitsfile *fptr,int subint,long double timeFromStart);
void removeTables(fitsfile *fptr,controlStruct *control);
void writeHeaderParameters(fitsfile *fptr,controlStruct *control);
void runTempo2(controlStruct *control);
void writePredictor(fitsfile *fptr,char *fname);
void writeEphemeris(fitsfile *fptr,controlStruct *control);
void calculatePeriod(controlStruct *control, T2Predictor pred, long double timeFromStart);
void calculateStt_offs(controlStruct *control, T2Predictor pred);
//double evaluateTemplate(controlStruct *control,int chan,int pol,int bin);
void calculatePhaseOffset(int chan,controlStruct *control,T2Predictor pred,long double timeFromStart);
int readObservation(FILE *fin,controlStruct *control);
void initialiseControl(controlStruct *control);
//double calculateScintScale(controlStruct *control,int chan,long double timeFromStart,int sub,long int *seed,float **scint,int *setScint);
