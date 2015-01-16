#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <fftw3.h>

#include "psim.h"
#include "T2toolkit.h"
#include "fitsio.h"
#include "tempo2pred.h"

int main(int argc,char *argv[])
{
  T2Predictor pred;
  channel *chan;
  controlStruct control;
  int nPhaseBins=1024;
  int i,j,k,l;
  int ret;
  long seed;
  int sub;
  long double timeFromStart;
  
	double tempFlux;

  fitsfile *fptr;
  int status=0;
  char file[128];

  char obsTable[128];
  char pTemplate[128];
  int  endit=0;
  FILE *fin;

  tmplStruct tmpl;
  double scintScale=1;
	acfStruct acfStructure; // dynamical spectrum
  //float **scint;
  //int setScint=0;
  //int nx=16384;
  //int ny=2048;
  //int ny=1024;
  
  //scint = (float **)malloc(sizeof(float *)*ny);
  //for (i=0;i<ny;i++)
  //  scint[i] = (float *)malloc(sizeof(float)*nx);      
  
  seed = TKsetSeed();

  for (i=0;i<argc;i++)
	{
		if (strcmp(argv[i],"-p")==0) // Input parameters
			strcpy(obsTable,argv[++i]);
	}
  
  if (!(fin = fopen(obsTable,"r")))
  {
    printf("Unable to open observation table: %s\n",obsTable);
    exit(1);
  }

  initialiseTemplate(&tmpl);
  initialiseControl(&control);

  while (endit==0)
	{
		endit = readObservation(fin,&control);
   
		readTemplate(control.template,&tmpl);
    
		if (endit==1)
		{
			printf("All input read\n");
			break;
		}
    
		puts(control.fname);
		// Allocate memory
   
		control.phaseOffset = (long double *)malloc(sizeof(long double)*control.nchan); 
    chan = (channel *)malloc(sizeof(channel)*control.nchan);
    for (i=0;i<control.nchan;i++)
		{
			chan[i].npol = control.npol;
	  	chan[i].pol = (polarisation *)malloc(sizeof(polarisation)*control.npol);
	  	for (j=0;j<control.npol;j++)
	  	  chan[i].pol[j].val = (double *)malloc(sizeof(double)*nPhaseBins);
		}
      
		sprintf(file,"!%s(psrheader.fits)",control.fname);
		fits_create_file(&fptr,file,&status);
    fits_report_error(stdout,status);
    createFitsFile(fptr,&control);
    writeEphemeris(fptr,&control);
      
    // Create predictor
    T2Predictor_Init(&pred);
    runTempo2(&control);
    writePredictor(fptr,"t2pred.dat");
    if (ret=T2Predictor_Read(&pred,(char *)"t2pred.dat"))
		{
			printf("Error: unable to read predictor\n");
			exit(1);
		}
      
		// calculate the dynamic spectrum

		if (control.scint_ts > 0)
		{
			calculateScintScale (&acfStructure, &control);
		}

		//////////////////////////////////////////////////////////////
    timeFromStart = 0;
    for (i=0;i<control.nsub;i++)
		{
			// Calculate period for this subint
			calculatePeriod(i,&control,pred,timeFromStart);

			// Calculate phase offset for each channel
	  	for (j=0;j<control.nchan;j++)
	    {
				calculatePhaseOffset(j,&control,pred,timeFromStart);
			}

	  	//printf("period = %g\n",(double)control.period);
			#pragma omp parallel for private(k,l,scintScale,tempFlux)
	  	for (j=0;j<control.nchan;j++)
	    {
				if (control.scint_ts > 0)
				{
					//printf("Here with timeFromStart %Lg\n",timeFromStart);
					scintScale = acfStructure.dynSpecWindow[j][i];
				}
	      
				tempFlux = 0.0;
				for (l=0;l<nPhaseBins;l++)
				{
					tempFlux += evaluateTemplate(&control,&tmpl,j,0,l);
				}
				tempFlux = tempFlux/nPhaseBins;

				for (k=0;k<control.npol;k++)
				{
					for (l=0;l<nPhaseBins;l++)
					{
						//chan[j].pol[k].val[l] = control.whiteLevel*TKgaussDev(&seed);
		      	//chan[j].pol[k].val[l] += scintScale*evaluateTemplate(&control,&tmpl,j,k,l);
		      	chan[j].pol[k].val[l] = scintScale*evaluateTemplate(&control,&tmpl,j,k,l)*control.flux[j]/tempFlux + control.radioNoise*TKgaussDev(&seed);
		      	//printf ("%lf\n",evaluateTemplate(&control,&tmpl,j,k,l));
					}
				}
			}
	  
			writeChannels(chan,&control,fptr,i+1,timeFromStart);
			timeFromStart += control.tsub;
		}
		fits_close_file(fptr,&status);
    T2Predictor_Destroy(&pred);

    // Deallocate memory
    free(control.phaseOffset);
    for (i=0;i<control.nchan;i++)
		{
			for (j=0;j<control.npol;j++)
				free(chan[i].pol[j].val);
			free(chan[i].pol);
		}
      
		free(chan);
	}
  
	deallocateMemory (&acfStructure);
	fclose(fin);
  // Should deallocate the memory
}

void createFitsFile(fitsfile *fptr,controlStruct *control)
{
  removeTables(fptr,control);
  writeHeaderParameters(fptr,control);
}

void writeHeaderParameters(fitsfile *fptr,controlStruct *control)
{
  FILE *fin;
  char keyword[128],setVal[128];
  int status=0;
  double obsBW;

  fits_movabs_hdu( fptr, 1, NULL, &status );
  fits_write_date(fptr, &status);
  
  if (!(fin = fopen(control->primaryHeaderParams,"r")))
    {
      printf("Error: Unable to open file >%s<\n",control->primaryHeaderParams);
      exit(1);
    }

  while (!feof(fin))
    {
      if (fscanf(fin,"%s %s",keyword,setVal)==2)
	{
	  //	  printf("Read: %s %s\n",keyword,setVal);
	  fits_update_key(fptr, TSTRING, keyword, setVal, NULL, &status );
	  fits_report_error(stdout,status);
	}
    }
  fits_update_key(fptr, TSTRING, (char *)"OBS_MODE", (char *)"PSR", NULL, &status );
  fits_update_key(fptr, TSTRING, (char *)"CAL_MODE", (char *)"OFF", NULL, &status );
  fits_report_error(stdout,status);

  //    fits_update_key(fptr,TSTRING, (char *)"SRC_NAME",&(data->srcName),NULL,&status);
  fits_update_key(fptr,TDOUBLE, (char *)"OBSFREQ",&(control->cFreq),NULL,&status);
  obsBW = fabs(control->obsBW);
  fits_update_key(fptr,TDOUBLE, (char *)"OBSBW",&(obsBW),NULL,&status);
  fits_update_key(fptr,TINT, (char *)"OBSNCHAN",&(control->nchan),NULL,&status);
  //    fits_update_key(fptr,TDOUBLE, (char *)"SCANLEN",&(data->scanLength),NULL,&status);
      
  fits_update_key(fptr,TINT, (char *)"STT_IMJD",&(control->stt_imjd),NULL,&status);
  fits_update_key(fptr,TDOUBLE, (char *)"STT_SMJD",&(control->stt_smjd),NULL,&status);
  fits_update_key(fptr,TDOUBLE, (char *)"STT_OFFS",&(control->stt_offs),NULL,&status); 
  fclose(fin);

  // Now calculate the Local Sidereal Time (LST)
  /*  mjd = data->stt_imjd + (data->stt_smjd + data->stt_offs)/86400.0L;
      lst = (double)calcLocalSiderealTime(mjd,data)*60.0*60.0;
      fits_update_key(fptr,TDOUBLE, (char *)"STT_LST",&lst,NULL,&status); */
      
  // SHOULD FIX THIS
  // HARDCODE FOR 1022
  /*  {
  //    char raj[128] = "04:37:15.810";
  //    char decj[128] = "-47:15:08.600";
  char raj[128] = "10:22:15.810";
  char decj[128] = "+10:01:08.600";
  fits_update_key(fptr, TSTRING, (char *)"RA", &raj, NULL, &status );
  fits_update_key(fptr, TSTRING, (char *)"DEC", &decj, NULL, &status );
  } */

}

void removeTables(fitsfile *fptr,controlStruct *control)
{
  int newHDUtype;
  int status=0;

  fits_movnam_hdu(fptr, BINARY_TBL, "FLUX_CAL", 0, &status);  fits_delete_hdu(fptr, &newHDUtype, &status);
  fits_movnam_hdu(fptr, BINARY_TBL, "COHDDISP", 0, &status);  fits_delete_hdu(fptr, &newHDUtype, &status);
  fits_movnam_hdu(fptr, BINARY_TBL, "POLYCO", 0, &status);  fits_delete_hdu(fptr, &newHDUtype, &status);
  fits_movnam_hdu(fptr, BINARY_TBL, "CAL_POLN", 0, &status);  fits_delete_hdu(fptr, &newHDUtype, &status);
  fits_movnam_hdu(fptr, BINARY_TBL, "FEEDPAR", 0, &status);  fits_delete_hdu(fptr, &newHDUtype, &status);
  if (status)
    {
      fits_report_error(stdout,status);
      exit(1);
    }
}

void writeChannels(channel *chan,controlStruct *control,fitsfile *fptr,int subint,long double timeFromStart)
{
  int status=0;
  int dval_0=0;
  int dval_1=1;
  int colnum;
  double tbin = (double)(control->period/control->nbin);
  double chan_bw = (control->obsBW/control->nchan); 

  fits_movnam_hdu(fptr,BINARY_TBL,(char *)"SUBINT",0,&status);
  if (status) {fits_report_error(stdout,status); exit(1);}
  fits_update_key(fptr, TINT, (char *)"NAXIS2", &subint, NULL, &status );
  if (status) {fits_report_error(stdout,status); exit(1);}
  
  if (subint==1)
    {
      fits_update_key(fptr, TSTRING, (char *)"INT_TYPE", (char *)"TIME", NULL, &status );
      fits_update_key(fptr, TSTRING, (char *)"INT_UNIT", (char *)"SEC", NULL, &status );
      fits_update_key(fptr, TSTRING, (char *)"SCALE", (char *)"FluxDen", NULL, &status );
      fits_update_key(fptr, TSTRING, (char *)"POL_TYPE", (char *)"INTEN", NULL, &status );
      fits_update_key(fptr,TINT, (char *)"NPOL",&(control->npol),NULL,&status);  
      fits_update_key(fptr,TINT, (char *)"NBIN",&(control->nbin),NULL,&status);  
      fits_update_key(fptr,TINT, (char *)"NBIN_PRD",&(control->nbin),NULL,&status);  
      fits_update_key(fptr,TINT, (char *)"PHS_OFFS",&dval_0,NULL,&status);  
      fits_update_key(fptr,TINT, (char *)"NBITS",&dval_1,NULL,&status);  
      fits_update_key(fptr,TINT, (char *)"ZERO_OFF",&dval_0,NULL,&status);  
      fits_update_key(fptr,TINT, (char *)"NSUBOFFS",&dval_0,NULL,&status);  
      fits_update_key(fptr,TINT, (char *)"NCHAN",&(control->nchan),NULL,&status);  
      fits_update_key(fptr,TDOUBLE, (char *)"CHAN_BW",&chan_bw,NULL,&status);  
      fits_update_key(fptr,TDOUBLE, (char *)"TBIN",&tbin,NULL,&status);  
      fits_update_key(fptr,TDOUBLE, (char *)"DM",&(control->dm),NULL,&status);  
      fits_update_key(fptr,TINT, (char *)"RM",&dval_0,NULL,&status);  
      fits_update_key(fptr,TINT, (char *)"NCHNOFFS",&dval_0,NULL,&status);  
      fits_update_key(fptr,TINT, (char *)"NSBLK",&dval_1,NULL,&status);  
      // We need to understand what this parameter really is
      //      fits_update_key(fptr,TSTRING, (char *)"EPOCHS",(char *)"VALID",NULL,&status);  
    }
  // Now write the information for this new subint
  {
    int indxval = 0;
    int nchan = control->nchan;
    int npol  = control->npol;
    int nbin  = control->nbin;
    long naxes[4];
    int naxis=3;
    float dat_freq[nchan],dat_wts[nchan],dat_offs[nchan*npol],dat_scl[nchan*npol];
    double rajd,decjd;
    double lst_sub,para;
    long double mjd;
    double f0;
    double tsub = control->tsub;
    double offsSub = (double)timeFromStart+(int)(control->tsub/2.0/control->period+0.5)*control->period; // This should be offset from start of subint centre
    //double offsSub = (double)timeFromStart+tsub/2.0; // This should be offset from start of subint centre
    int i,j,n;
    float dataVals[nchan*nbin];

    // MUST FIX
        rajd = 69.3158537729834; // Hardcoded to 0437
        decjd = -47.2523961499288; // Hardcoded to 0437
    //    rajd = 155.74168; // Hardcoded to 1022
    //    decjd = 10.03132; // Hardcoded to 1022

    // Now calculate the Local Sidereal Time (LST)
    // MUST SET
    //    mjd = data->stt_imjd + (data->stt_smjd + data->stt_offs)/86400.0L + subint*tsub/86400.0;
    //    lst_sub = (double)calcLocalSiderealTime(mjd,data)*60.0*60.0;
    //    fits_get_colnum(fptr,CASEINSEN,"LST_SUB",&colnum,&status);
    //    fits_write_col(fptr,TDOUBLE,colnum,subint,1,1,&lst_sub,&status);

    fits_get_colnum(fptr,CASEINSEN,"TSUBINT",&colnum,&status);
    fits_write_col(fptr,TDOUBLE,colnum,subint,1,1,&tsub,&status);

    fits_get_colnum(fptr,CASEINSEN,"OFFS_SUB",&colnum,&status);
    fits_write_col(fptr,TDOUBLE,colnum,subint,1,1,&offsSub,&status);

    fits_get_colnum(fptr,CASEINSEN,"RA_SUB",&colnum,&status);
    fits_write_col(fptr,TDOUBLE,colnum,subint,1,1,&rajd,&status);

    fits_get_colnum(fptr,CASEINSEN,"DEC_SUB",&colnum,&status);
    fits_write_col(fptr,TDOUBLE,colnum,subint,1,1,&decjd,&status);


    // NOTE THAT THIS SHOULD BE THE PAR_ANG AT THE SUBINT CENTRE -- CURRENTLY AT THE START
    //
    //    para = calculatePara(data);
    //    fits_get_colnum(fptr,CASEINSEN,"PAR_ANG",&colnum,&status);
    //    fits_write_col(fptr,TDOUBLE,colnum,subint,1,1,&para,&status);

    // MUST CHECK IF THE POS ANGLE SHOULD BE INDENTICAL TO THE PAR_ANG
    //    fits_get_colnum(fptr,CASEINSEN,"POS_ANG",&colnum,&status);
    //    fits_write_col(fptr,TDOUBLE,colnum,subint,1,1,&para,&status);

  f0 = control->cFreq + fabs(control->obsBW)/2.0; // Highest frequency
    
    for (i=0;i<nchan;i++)
      {
	//dat_freq[i] = f0-fabs(control->obsBW/(double)control->nchan)*i+fabs(control->obsBW/(double)control->nchan)*0.5; // Must fix
	dat_freq[i] = f0-fabs(control->obsBW/(double)control->nchan)*i; // Must fix
	dat_wts[i] = 1;
      }
    for (i=0;i<nchan*npol;i++)
      {
	dat_offs[i] = 0;
	dat_scl[i] = 1;
      }
    
    fits_get_colnum(fptr,CASEINSEN,"INDEXVAL",&colnum,&status);
    fits_report_error(stdout,status);
    fits_write_col(fptr,TINT,colnum,subint,1,1,&indxval,&status);
    fits_report_error(stdout,status); 

    // Write the data
    fits_get_colnum(fptr, CASEINSEN, "DATA", &colnum, &status);  
    fits_modify_vector_len (fptr, colnum, (nchan*npol*nbin), &status); 
    if (status) {fits_report_error(stdout,status); exit(1);}

    naxes[0] = nbin;
    naxes[1] = nchan;
    naxes[2] = npol;
    fits_delete_key(fptr, "TDIM18", &status); // THIS SHOULD NOT BE HARDCODED
    fits_write_tdim(fptr, colnum, naxis, naxes, &status);
    
    fits_get_colnum(fptr, CASEINSEN, "DATA", &colnum, &status);
    // Calculate scaling parameters
    {
      double i_mean,i_min,i_max;
      double scaleI,offsI;
          for (j=0;j<nchan;j++)
	{
	  i_mean = 0.0;
	  for (i=0;i<nbin;i++)
	    {
	      if (i==0)
		i_min = i_max = chan[j].pol[0].val[i];
	      else
		{
		  if (i_min > chan[j].pol[0].val[i]) i_min = chan[j].pol[0].val[i];
		  if (i_max < chan[j].pol[0].val[i]) i_max = chan[j].pol[0].val[i];
		}
	      i_mean+=chan[j].pol[0].val[i];
	    }
	  scaleI = (i_max-i_min)/2.0/16384.0;
	  offsI = i_max - 16384.0*scaleI;
	  for (i=0;i<nbin;i++)
	    chan[j].pol[0].val[i] = (chan[j].pol[0].val[i] - offsI)/scaleI;
	  dat_scl[j] = scaleI; dat_offs[j]   = (offsI*i_mean)/(double)nbin;
	}
    }
      //      printf("Writing %g\n",data->aa[i]);
    n=0;
    for (j=0;j<nchan;j++)
      {
	for (i=0;i<nbin;i++)
	  dataVals[n++] = (float)(chan[j].pol[0].val[i]);
      }
    printf("subint = %d, colnum = %d\n",subint,colnum);
    fits_write_col(fptr,TFLOAT,colnum,subint,1,n,dataVals,&status);
     
    fits_get_colnum(fptr, CASEINSEN, "DAT_FREQ", &colnum, &status);
    fits_modify_vector_len (fptr, colnum, nchan, &status); 
    fits_write_col(fptr,TFLOAT,colnum,subint,1,nchan,dat_freq,&status);
    fits_get_colnum(fptr, CASEINSEN, "DAT_WTS", &colnum, &status);
    fits_modify_vector_len (fptr, colnum, nchan, &status); 
    fits_write_col(fptr,TFLOAT,colnum,subint,1,nchan,dat_wts,&status);

    fits_get_colnum(fptr, CASEINSEN, "DAT_OFFS", &colnum, &status);
    fits_modify_vector_len (fptr, colnum, nchan*npol, &status); 
    fits_write_col(fptr,TFLOAT,colnum,subint,1,nchan*npol,dat_offs,&status);
    fits_get_colnum(fptr, CASEINSEN, "DAT_SCL", &colnum, &status);
    fits_modify_vector_len (fptr, colnum, nchan*npol, &status); 
    fits_write_col(fptr,TFLOAT,colnum,subint,1,nchan*npol,dat_scl,&status);


  }
  
}

void runTempo2(controlStruct *control)
{
  char execString[1024];
  double seg_length = control->segLength;
  int nfreqcoeff = control->nfreqcoeff;
  int ntimecoeff = control->ntimecoeff;
  printf("nfreqCoeff: %d\n", nfreqcoeff);

  double freq1 = control->cFreq - fabs(control->obsBW); 
  double freq2 = control->cFreq + fabs(control->obsBW);
  long double mjd1 = control->stt_imjd + control->stt_smjd/86400.0L - 60*60/86400.0L; // MUST FIX
  long double mjd2 = control->stt_imjd + control->stt_smjd/86400.0L + (control->nsub)*control->tsubRequested/86400.0 + 60*60/86400.0L; // MUST FIX

  //  sprintf(execString,"tempo2 -pred \"PKS %Lf %Lf %g %g %d %d %g\" -f %s",mjd1,mjd2,freq1,freq2,ntimecoeff,nfreqcoeff,seg_length,control->exact_ephemeris);
  sprintf(execString,"tempo2 -pred \"%s %Lf %Lf %g %g %d %d %g\" -f %s","PKS",mjd1,mjd2,freq1,freq2,ntimecoeff,nfreqcoeff,seg_length,control->exact_ephemeris);
  //sprintf(execString,"tempo2 -pred \"%s %Lf %Lf %g %g %d %d %g\" -f %s","BAT",mjd1,mjd2,freq1,freq2,ntimecoeff,nfreqcoeff,seg_length,control->exact_ephemeris);
  //sprintf(execString,"tempo2-dev -pred \"PKS %Lf %Lf %g %g %d %d %g\" -f %s",mjd1,mjd2,freq1,freq2,ntimecoeff,nfreqcoeff,seg_length,control->exact_ephemeris);
  printf("Running tempo2 to get predictor\n");
  system(execString);
  printf("Complete running tempo2\n");
}

void writeEphemeris(fitsfile *fptr,controlStruct *control)
{
  FILE *fin;
  char line[128];
  int colnum;
  int arr=1;
  int status=0;
  int linenum=41;
  char word1[128],word2[128];
  char *temp = &(line[0]);
  fits_movnam_hdu(fptr,BINARY_TBL,(char *)"PSRPARAM",0,&status);
  if (status) {fits_report_error(stdout,status); exit(1);}
  linenum=1;
  fits_get_colnum(fptr,CASEINSEN,"PARAM",&colnum,&status);
  fits_report_error(stdout,status);

  if (!(fin = fopen(control->exact_ephemeris,"r")))
    {
      printf("Unable to open file >%s<\n",control->exact_ephemeris);
      exit(1);      
    }
  while (!feof(fin))
    {
      if (fgets(line,128,fin))
	{
	  line[strlen(line)-1]='\0';
	  fits_write_col(fptr,TSTRING,colnum,linenum++,1,1,&temp,&status);
	  if (status) {fits_report_error(stdout,status);  exit(1);}
	  if (sscanf(line,"%s %s",word1,word2)==2)
	    {
	      if (strcasecmp(word1,"DM")==0)
		sscanf(word2,"%lf",&(control->dm));
	    }
	}
    }
  fclose(fin);
}

void writePredictor(fitsfile *fptr,char *fname)
{
  FILE *fin;
  char line[128];
  int colnum;
  int arr=1;
  int status=0;
  int linenum=41;
  char *temp = &(line[0]);
  fits_movnam_hdu(fptr,BINARY_TBL,(char *)"T2PREDICT",0,&status);
  if (status) {fits_report_error(stdout,status); exit(1);}
  linenum=1;
  fits_get_colnum(fptr,CASEINSEN,"PREDICT",&colnum,&status);
  fits_report_error(stdout,status);

  if (!(fin = fopen(fname,"r")))
    {
      printf("Unable to open file >%s<\n",fname);
      exit(1);      
    }
  while (!feof(fin))
    {
      if (fgets(line,128,fin))
	{
	  line[strlen(line)-1]='\0';
	  fits_write_col(fptr,TSTRING,colnum,linenum++,1,1,&temp,&status);
	  if (status) {fits_report_error(stdout,status);  exit(1);}
	}
    }
  fclose(fin);
}

void calculatePeriod(int sub,controlStruct *control, T2Predictor pred, long double timeFromStart)
{
  long double mjd0;
  long double freq;
  long double toff;

	/*
  freq = control->cFreq;
  toff = (int)(control->tsubRequested/2.0/control->period+0.5)*control->period;
  mjd0 = control->stt_imjd + (control->stt_smjd + control->stt_offs)/86400.0L + (timeFromStart + toff)/86400.0L; // Centre of subint
		//		mjd0 = control->stt_imjd + (control->stt_smjd + control->stt_offs + timeFromStart)/86400.0L + control->tsubRequested*0.5/86400.0L;
  control->period = 1.0/T2Predictor_GetFrequency(&pred,mjd0,freq);
  control->tsub = ((int)(control->tsubRequested/control->period+0.5))*control->period;
	printf("Here with %g %g %g %g\n",(double)freq,(double)mjd0,(double)control->period,(double)control->tsub);

  toff = (int)(control->tsub/2.0/control->period+0.5)*control->period;
  mjd0 = control->stt_imjd + (control->stt_smjd + control->stt_offs)/86400.0L + (timeFromStart + toff)/86400.0L; // Centre of subint
		//		mjd0 = control->stt_imjd + (control->stt_smjd + control->stt_offs + timeFromStart)/86400.0L + control->tsubRequested*0.5/86400.0L;
  control->period = 1.0/T2Predictor_GetFrequency(&pred,mjd0,freq);
  control->tsub = ((int)(control->tsubRequested/control->period+0.5))*control->period;

	printf("Here2 with %g %g %g %g\n",(double)freq,(double)mjd0,(double)control->period,(double)control->tsub);
	*/
	if (sub == 0)
	{
		freq = control->cFreq;
		mjd0 = control->stt_imjd + (control->stt_smjd + control->stt_offs)/86400.0L + control->tsubRequested*0.5/86400.0L;
		control->period = 1.0/T2Predictor_GetFrequency(&pred,mjd0,freq);
		control->tsub = ((int)(control->tsubRequested/control->period+0.5))*control->period;
		//printf("Here with %g %g %g %g\n",(double)freq,(double)mjd0,(double)control->period,(double)control->tsub);

		//Iterate once more with tsub instead of tsubRequested
		mjd0 = control->stt_imjd + (control->stt_smjd + control->stt_offs)/86400.0L + control->tsub*0.5/86400.0L;
		control->period = 1.0/T2Predictor_GetFrequency(&pred,mjd0,freq);
		control->tsub = ((int)(control->tsubRequested/control->period+0.5))*control->period;
		//printf("Here2 with %g %g %g %g\n",(double)freq,(double)mjd0,(double)control->period,(double)control->tsub);
	}
	else
	{
		freq = control->cFreq;
		//toff = 0.0;
		toff = (int)(control->tsub/2.0/control->period+0.5)*control->period;
		mjd0 = control->stt_imjd + (control->stt_smjd + control->stt_offs)/86400.0L + (timeFromStart + toff)/86400.0L;
		control->period = 1.0/T2Predictor_GetFrequency(&pred,mjd0,freq);
		control->tsub = ((int)(control->tsubRequested/control->period+0.5))*control->period;
		//printf("Here3 with %g %g %g %g\n",(double)freq,(double)mjd0,(double)control->period,(double)control->tsub);

		//toff = 0.0;
		toff = (int)(control->tsub/2.0/control->period+0.5)*control->period;
		mjd0 = control->stt_imjd + (control->stt_smjd + control->stt_offs)/86400.0L + (timeFromStart + toff)/86400.0L;
		control->period = 1.0/T2Predictor_GetFrequency(&pred,mjd0,freq);
		control->tsub = ((int)(control->tsubRequested/control->period+0.5))*control->period;
		//printf("Here4 with %g %g %g %g\n",(double)freq,(double)mjd0,(double)control->period,(double)control->tsub);

		//mjd0 = control->stt_imjd + (control->stt_smjd + control->stt_offs)/86400.0L + (timeFromStart + toff)/86400.0L;
		} 
}

// Initialises the template, but does not allocate memory for the profiles
// This routine should be called at the start of the program
void initialiseTemplate(tmplStruct *tmpl)
{
  strcpy(tmpl->dte,"UNSET");
  strcpy(tmpl->user,"UNSET");
  tmpl->templateVersion = 0;
  strcpy(tmpl->source,"UNSET");
  strcpy(tmpl->profileFile,"UNSET");
  strcpy(tmpl->units,"UNSET");
  tmpl->dedispersed = 0;
  tmpl->nchan = 0;
  tmpl->channelMemoryAllocated=0;
  tmpl->nChannelAllocated=0;
}


// Reads a template from disk
void readTemplate(char *file,tmplStruct *tmpl)
{
  FILE *fin;
  int i;
  char line[4096];
  char firstword[4096];
  char dummy[4096];
  int nchan=-1;
  int nstokes=1;
  int chan,stokes,comp,ivm,icomp;
  char stokesStr[1024];
  double f1,f2;

  // Read primary header
  if (!(fin = fopen(file,"r"))){
    printf("Unable to open file: >%s<\n",file);
    exit(1);
  }
  while (!feof(fin))
    {
      if (fgets(line,4096,fin) != NULL){
	if (line[0] == '#') // Comment line
	  {
	    // Do nothing
	  }
	else {
	  sscanf(line,"%s",firstword);
	  if (strcasecmp(firstword,"TEMPLATE_VERSION:")==0)
		{
	    sscanf(line,"%s %f",dummy,&(tmpl->templateVersion));
			//printf("%f\n", dummy);
		}
	  else if (strcasecmp(firstword,"SOURCE:")==0)
	    sscanf(line,"%s %s",dummy,(tmpl->source));
	  else if (strcasecmp(firstword,"PROFILE_FILE:")==0)
	    sscanf(line,"%s %s",dummy,(tmpl->profileFile));
	  else if (strcasecmp(firstword,"DATE:")==0)
	    sscanf(line,"%s %s",dummy,(tmpl->dte));
	  else if (strcasecmp(firstword,"UNITS:")==0)
	    sscanf(line,"%s %s",dummy,(tmpl->units));
	  else if (strcasecmp(firstword,"ID:")==0)
	    sscanf(line,"%s %s",dummy,(tmpl->user));
	  else if (strcasecmp(firstword,"DM_CORRECTION:")==0)
	    sscanf(line,"%s %lf",dummy,&(tmpl->dedispersed));
	  else if (strcasecmp(firstword,"NCHAN:")==0)
	    sscanf(line,"%s %d",dummy,&nchan);
	  else if (strcasecmp(firstword,"STOKES:")==0)
	    {
	      sscanf(line,"%s %s",dummy,stokesStr);
	      if (strcmp(stokesStr,"I")==0)
		nstokes=1;
	      else if (strcmp(stokesStr,"Q")==0 || strcmp(stokesStr,"U")==0 || strcmp(stokesStr,"V")==0)
		nstokes=4;
	    }
	}
      }
    }
  fclose(fin);
  // Do some checks
  if (nchan < 0){
    printf("Have not defined any channels. Unable to continue\n");
    exit(1);
  }
  // Allocate memory for these channels
  tmpl->nchan = nchan;
  if (tmpl->channelMemoryAllocated == 0){
    if (!(tmpl->channel = (channelStruct *)malloc(sizeof(channelStruct)*nchan))){
      printf("ERROR in allocated memory for channels\n");
      exit(1);
    }
    printf("Allocated %d channels\n",nchan);
    tmpl->channelMemoryAllocated = 1;
    tmpl->nChannelAllocated = nchan;
    for (i=0;i<nchan;i++)
      tmpl->channel[i].polMemoryAllocated = 0;
  }

  chan = -1;
  stokes = -1;
  comp=-1;
  // Now read the data
  if (!(fin = fopen(file,"r"))){
    printf("Unable to open file: >%s<\n",file);
    exit(1);
  }
  while (!feof(fin))
    {
      if (fgets(line,4096,fin)!=NULL)
			{
				if (line[0] == '#') // Comment line
				{
					// Do nothing
				}
				else 
				{
					sscanf(line,"%s",firstword);
					if (strcasecmp(firstword,"STOKES:")==0)
					{
						sscanf(line,"%s %s",dummy,stokesStr);
						if (strcmp(stokesStr,"I")==0)
							stokes=0;
						else if (strcmp(stokesStr,"Q")==0)
							stokes = 1;
						else if (strcmp(stokesStr,"U")==0)
							stokes = 2;
						else if (strcmp(stokesStr,"V")==0)
							stokes = 3;
					} 
					else if (strcasecmp(firstword,"FREQUENCY_RANGE:")==0)
					{
						if (stokes==0)
							chan++;
						sscanf(line,"%s %lf %lf",dummy,&f1,&f2);
						if (f1 < f2)
						{
							tmpl->channel[chan].freqLow = f1;
							tmpl->channel[chan].freqHigh = f2;
							//printf("%lf %lf\n",tmpl->channel[chan].freqLow,tmpl->channel[chan].freqHigh);
						} 
						else 
						{
							tmpl->channel[chan].freqLow = f2;
							tmpl->channel[chan].freqHigh = f1;
						}
						tmpl->channel[chan].nstokes = nstokes;
						// Allocate memory
						if (tmpl->channel[chan].polMemoryAllocated==0)
						{
							if (!(tmpl->channel[chan].pol = (polStruct *)malloc(sizeof(polStruct)*nstokes)))
							{
								printf("Error in allocated memory for Stokes\n");
								exit(1);
							}
		
							tmpl->channel[chan].polMemoryAllocated = 1;
							for (i=0;i<nstokes;i++)
								tmpl->channel[chan].pol[i].compMemoryAllocated = 0;
						}
	      //	    }
						tmpl->channel[chan].nPolAllocated = nstokes;
				}
				/*
				// using group of Von Mises
				else if (strcasecmp(firstword,"NCOMP:")==0)
				{
					int ncomp;
					sscanf(line,"%s %d",dummy,&ncomp);
					tmpl->channel[chan].pol[stokes].nComp = ncomp;
					tmpl->channel[chan].pol[stokes].stokes = stokes;
					if (tmpl->channel[chan].pol[stokes].compMemoryAllocated==0)
					{
						if (!(tmpl->channel[chan].pol[stokes].comp = (component *)malloc(sizeof(component)*ncomp)))
						{
							printf("Error in allocated memory for components\n");
							exit(1);
						}
		
						tmpl->channel[chan].pol[stokes].compMemoryAllocated = 1;
					}
					tmpl->channel[chan].pol[stokes].nCompAllocated = ncomp;
					//printf ("%d\n",ncomp);
				}
				else if (strcasecmp(firstword,"NVonMises:")==0)
				{
					int nAllVm;
					sscanf(line,"%s %d",dummy,&nAllVm);
					tmpl->channel[chan].pol[stokes].allVm = nAllVm;
					//printf ("%d\n",nAllVm);
	      
					comp=0;
					icomp=0;
					ivm=0;
				}
				else // Look for the number of Von Mises for each component
				{
					char substr[4096];
					strcpy(substr,firstword);
					substr[4]='\0';
	      
					if (strcasecmp(substr,"VonM")==0)
					{
						//printf ("%d\n",comp);
						tmpl->channel[chan].pol[stokes].comp[comp].vmMemoryAllocated=0;
						sscanf(line,"%s %d",dummy, &(tmpl->channel[chan].pol[stokes].comp[comp].nVm));
						//printf ("COMP%d has %d Von Mises functions\n",comp+1, tmpl->channel[chan].pol[stokes].comp[comp].nVm);
						if (tmpl->channel[chan].pol[stokes].comp[comp].vmMemoryAllocated==0)
						{
							if (!(tmpl->channel[chan].pol[stokes].comp[comp].vonMises = (vMises *)malloc(sizeof(vMises)*tmpl->channel[chan].pol[stokes].comp[comp].nVm)))
							{
								printf("Error in allocated memory for components\n");
								exit(1);
							}
							tmpl->channel[chan].pol[stokes].comp[comp].vmMemoryAllocated = 1;
							//printf ("%d\n",tmpl->channel[chan].pol[stokes].comp[comp].vmMemoryAllocated);
						}
						tmpl->channel[chan].pol[stokes].comp[comp].nVmAllocated = tmpl->channel[chan].pol[stokes].comp[comp].nVm;
						comp++;
					}
					else if (strcasecmp(substr,"COMP")==0)
					{
						//printf ("%d\n",tmpl->channel[chan].pol[stokes].comp[icomp].nVm);
						if (ivm != tmpl->channel[chan].pol[stokes].comp[icomp].nVm-1)
						{
							sscanf(line,"%s %lf %lf %lf %lf %lf %lf",dummy,
									&(tmpl->channel[chan].pol[stokes].comp[icomp].vonMises[ivm].height),
									&(tmpl->channel[chan].pol[stokes].comp[icomp].vonMises[ivm].height_err),
									&(tmpl->channel[chan].pol[stokes].comp[icomp].vonMises[ivm].concentration),
									&(tmpl->channel[chan].pol[stokes].comp[icomp].vonMises[ivm].concentration_err),
									&(tmpl->channel[chan].pol[stokes].comp[icomp].vonMises[ivm].centroid),
									&(tmpl->channel[chan].pol[stokes].comp[icomp].vonMises[ivm].centroid_err));
							//printf("%lf %lf %lf\n",tmpl->channel[chan].pol[stokes].comp[icomp].vonMises[ivm].height,tmpl->channel[chan].pol[stokes].comp[icomp].vonMises[ivm].concentration,tmpl->channel[chan].pol[stokes].comp[icomp].vonMises[ivm].centroid);
							//printf ("%d %d\n",ivm, icomp);
							ivm++;
						}
						else 
						{
							sscanf(line,"%s %lf %lf %lf %lf %lf %lf",dummy,
									&(tmpl->channel[chan].pol[stokes].comp[icomp].vonMises[ivm].height),
									&(tmpl->channel[chan].pol[stokes].comp[icomp].vonMises[ivm].height_err),
									&(tmpl->channel[chan].pol[stokes].comp[icomp].vonMises[ivm].concentration),
									&(tmpl->channel[chan].pol[stokes].comp[icomp].vonMises[ivm].concentration_err),
									&(tmpl->channel[chan].pol[stokes].comp[icomp].vonMises[ivm].centroid),
									&(tmpl->channel[chan].pol[stokes].comp[icomp].vonMises[ivm].centroid_err));
							//printf ("%d %d\n",ivm,icomp);
							icomp++;
							ivm = 0;
						}
		  
						//&(tmpl->channel[chan].pol[stokes].comp[comp].height),
						//&(tmpl->channel[chan].pol[stokes].comp[comp].height_err),
						//&(tmpl->channel[chan].pol[stokes].comp[comp].concentration),
						//&(tmpl->channel[chan].pol[stokes].comp[comp].concentration_err),
						//&(tmpl->channel[chan].pol[stokes].comp[comp].centroid),
						//&(tmpl->channel[chan].pol[stokes].comp[comp].centroid_err));
					}
				}
				*/
				// not using group of Von Mises
				else if (strcasecmp(firstword,"NCOMP:")==0)
				//else if (strcasecmp(firstword,"NVonMises:")==0)
				{
					int nAllVm, ncomp;
					sscanf(line,"%s %d",dummy,&nAllVm);
					ncomp = nAllVm;
					tmpl->channel[chan].pol[stokes].allVm = nAllVm;
					tmpl->channel[chan].pol[stokes].stokes = stokes;
					tmpl->channel[chan].pol[stokes].nComp = ncomp;    

					if (tmpl->channel[chan].pol[stokes].compMemoryAllocated==0)
					{
						if (!(tmpl->channel[chan].pol[stokes].comp = (component *)malloc(sizeof(component)*ncomp)))
						{
							printf("Error in allocated memory for components\n");
							exit(1);
						}
		
						tmpl->channel[chan].pol[stokes].compMemoryAllocated = 1;
					}
					tmpl->channel[chan].pol[stokes].nCompAllocated = ncomp;

					for (comp = 0; comp < tmpl->channel[chan].pol[stokes].nComp; comp++)
					{
						tmpl->channel[chan].pol[stokes].comp[comp].vmMemoryAllocated=0;
						tmpl->channel[chan].pol[stokes].comp[comp].nVm = 1;   

						if (tmpl->channel[chan].pol[stokes].comp[comp].vmMemoryAllocated==0)
						{
							if (!(tmpl->channel[chan].pol[stokes].comp[comp].vonMises = (vMises *)malloc(sizeof(vMises)*tmpl->channel[chan].pol[stokes].comp[comp].nVm)))
							{
								printf("Error in allocated memory for components\n");
								exit(1);
							}
							tmpl->channel[chan].pol[stokes].comp[comp].vmMemoryAllocated = 1;
							//printf ("%d\n",tmpl->channel[chan].pol[stokes].comp[comp].vmMemoryAllocated);
						}
						tmpl->channel[chan].pol[stokes].comp[comp].nVmAllocated = tmpl->channel[chan].pol[stokes].comp[comp].nVm;
					}

					icomp=0;
					ivm=0;
				}
				else // Look for each Von Mises 
				{
					char substr[4096];
					strcpy(substr,firstword);
					substr[4]='\0';
	      
					if (strcasecmp(substr,"COMP")==0)
					{
						sscanf(line,"%s %lf %lf %lf %lf %lf %lf",dummy,
									&(tmpl->channel[chan].pol[stokes].comp[icomp].vonMises[ivm].height),
									&(tmpl->channel[chan].pol[stokes].comp[icomp].vonMises[ivm].height_err),
									&(tmpl->channel[chan].pol[stokes].comp[icomp].vonMises[ivm].concentration),
									&(tmpl->channel[chan].pol[stokes].comp[icomp].vonMises[ivm].concentration_err),
									&(tmpl->channel[chan].pol[stokes].comp[icomp].vonMises[ivm].centroid),
									&(tmpl->channel[chan].pol[stokes].comp[icomp].vonMises[ivm].centroid_err));
							//printf ("%d %d\n",ivm,icomp);
						icomp++;
					}
				}
			}
		}
  }
  fclose(fin);
}


// Evaluate a single template component
double evaluateTemplateComponent(tmplStruct *tmpl,double phi,int chan,int stokes,int comp,double phiRot)
{
  double result=0;
  //result = tmpl->channel[chan].pol[stokes].comp[comp].height *
  int k;
  for (k=0;k<tmpl->channel[chan].pol[stokes].comp[comp].nVm;k++)
  	result += fabs(tmpl->channel[chan].pol[stokes].comp[comp].vonMises[k].height) *
    	exp(tmpl->channel[chan].pol[stokes].comp[comp].vonMises[k].concentration*
	(cos((phi - tmpl->channel[chan].pol[stokes].comp[comp].vonMises[k].centroid + phiRot)*2*M_PI)-1));
  return result;
}

/*
double evaluateTemplate(controlStruct *control,int chan,int pol,int bin)
{
  double val=0.0;
  double centre = 0.4;
  double height = 3;
  double conc = 40;
  val = height*exp(conc*(cos((bin/(double)control->nbin+control->phaseOffset[chan]-centre)*2*M_PI)-1));
  return val;
}
*/

// Evaluate a given frequency channel and polarisation
double evaluateTemplate(controlStruct *control, tmplStruct *tmpl, int chan, int pol, int bin)
{
  int tChan = (int)(chan*tmpl->nchan/control->nchan);

  double result=0;
  int k;
  for (k=0;k<tmpl->channel[tChan].pol[pol].nComp;k++)
    {
      result += evaluateTemplateComponent(tmpl,bin/(double)control->nbin,tChan,pol,k,control->phaseOffset[chan]);
    }
  return result;
}

void calculatePhaseOffset(int chan,controlStruct *control,T2Predictor pred,long double timeFromStart)
{
  long double f0,freq,phase0,mjd0;
  long double toff;

  //toff = (int)(control->tsub/2.0/control->period+0.5)*control->period;
	toff = 0.0;
  mjd0 = control->stt_imjd + (control->stt_smjd + control->stt_offs)/86400.0L + (timeFromStart + toff)/86400.0L;
  //mjd0 = control->stt_imjd + (control->stt_smjd + control->stt_offs)/86400.0L + (timeFromStart + control->tsub/2.0)/86400.0L;
  f0 = control->cFreq + fabs(control->obsBW)/2.0; // Highest frequency
  //freq = f0 - fabs(control->obsBW/(double)control->nchan)*chan + fabs(control->obsBW/(double)control->nchan)*0.5;
  freq = f0 - fabs(control->obsBW/(double)control->nchan)*chan;
  phase0 = T2Predictor_GetPhase(&pred,mjd0,freq);
  control->phaseOffset[chan] = (phase0 - floorl(phase0));
  //printf("DAI SHI: %.2Lf %.15Lf\n", freq, phase0);
  //printf("DAI SHI: %.2Lf %.15Lf\n", freq, control->phaseOffset[chan]);
  //printf("DAI SHI: %.15Lf %.15Lf\n",mjd0,control->phaseOffset[chan]);
}

int readObservation(FILE *fin,controlStruct *control)
{
  char param[1024],val[1024];
  int endit=-1;
  int finished=0;
	int i;
	double f0, f1;

  printf("Reading observation table\n");
  // Find the start observation
  while (!feof(fin))
	{
		if (fscanf(fin,"%s",param)==1)
		{
			if (strcasecmp(param,"START_OBS")==0)
	    {
	      endit=0;
	      break;
	    }
		}
		else 
			return 1;
	}

  printf("Got to this bit with %d\n",endit);
  if (endit==-1)
    return 1;

  do
	{
		fscanf(fin,"%s",param);
		if (strcasecmp(param,"END_OBS")==0)
			endit=1;
		else
		{
			if (strcasecmp(param,"PHEAD")==0)
				fscanf(fin,"%s",control->primaryHeaderParams);
			else if (strcasecmp(param,"SRC")==0)
			  fscanf(fin,"%s",control->src);
			else if (strcasecmp(param,"EXACT_EPHEMERIS")==0)
			  fscanf(fin,"%s",control->exact_ephemeris);
			else if (strcasecmp(param,"TEMPLATE")==0)
			  fscanf(fin,"%s",control->template);
			else if (strcasecmp(param,"SCINT_TS")==0)
			  fscanf(fin,"%lf",&(control->scint_ts));
			else if (strcasecmp(param,"SCINT_FREQBW")==0)
			  fscanf(fin,"%lf",&(control->scint_freqbw));	  
			else if (strcasecmp(param,"FILE")==0)
			  fscanf(fin,"%s",control->fname);
			else if (strcasecmp(param,"TYPE")==0)
			  fscanf(fin,"%s",control->type);
			else if (strcasecmp(param,"STT_IMJD")==0)
			  fscanf(fin,"%d",&(control->stt_imjd));
			else if (strcasecmp(param,"STT_SMJD")==0)
			  fscanf(fin,"%lf",&(control->stt_smjd));
			else if (strcasecmp(param,"STT_OFFS")==0)
			  fscanf(fin,"%lf",&(control->stt_offs));
			else if (strcasecmp(param,"TSUB")==0)
			  fscanf(fin,"%lf",&(control->tsubRequested));
			else if (strcasecmp(param,"CFREQ")==0)
			  fscanf(fin,"%lf",&(control->cFreq));
			else if (strcasecmp(param,"BW")==0)
			  fscanf(fin,"%lf",&(control->obsBW));
			else if (strcasecmp(param,"NCHAN")==0)
			  fscanf(fin,"%d",&(control->nchan));
			else if (strcasecmp(param,"NBIN")==0)
			  fscanf(fin,"%d",&(control->nbin));
			else if (strcasecmp(param,"NPOL")==0)
			  fscanf(fin,"%d",&(control->npol));
			else if (strcasecmp(param,"NSUB")==0)
			  fscanf(fin,"%d",&(control->nsub));
			else if (strcasecmp(param,"SEGLENGTH")==0)
			  fscanf(fin,"%lf",&(control->segLength));
			else if (strcasecmp(param,"NFREQ_COEFF")==0)
			  fscanf(fin,"%d",&(control->nfreqcoeff));
			else if (strcasecmp(param,"NTIME_COEFF")==0)
			  fscanf(fin,"%d",&(control->ntimecoeff));
			else if (strcasecmp(param,"WHITE_LEVEL")==0)
			  fscanf(fin,"%lf",&(control->whiteLevel));
			else if (strcasecmp(param,"TSYS")==0)
			  fscanf(fin,"%lf",&(control->tsys));
			else if (strcasecmp(param,"TSKY")==0)
			  fscanf(fin,"%lf",&(control->tsky));
			else if (strcasecmp(param,"GAIN")==0)
			  fscanf(fin,"%lf",&(control->gain));
			else if (strcasecmp(param,"CFLUX")==0)
			  fscanf(fin,"%lf",&(control->cFlux));
			else if (strcasecmp(param,"SI")==0)
				fscanf(fin,"%lf",&(control->si));
		}
	} while (endit==0);

	//////////////////////////////////////////////////////////////////////
  f0 = control->cFreq + fabs(control->obsBW)/2.0; // Highest frequency
	if (control->cFlux != 0.0 && control->si != 0.0)
	{
		control->flux = (double *)malloc(sizeof(double)*control->nchan);

		for (i = 0; i < control->nchan; i++)
		{
			f1 = f0 - fabs(control->obsBW/(double)control->nchan)*i;
			control->flux[i] = pow(f1/1369.0, control->si)*control->cFlux;
		}
	}
	else 
	{
		printf ("cFlux and spectral index needed!\n");
		exit (1);
	}

	//////////////////////////////////////////////////////////////////////
	if (control->tsys != 0.0 && control->tsky != 0.0 && control->gain != 0.0 && control->whiteLevel == 0)
	{
		control->radioNoise = (control->tsys+control->tsky)/(control->gain)/sqrt(2.0*(control->tsubRequested/control->nbin)*(fabs(control->obsBW)/control->nchan));
	}
	else if (control->tsys == 0.0 && control->tsky == 0.0 && control->gain == 0.0 && control->whiteLevel != 0)
	{
		control->radioNoise = control->whiteLevel;
	}
	else 
	{
		printf ("Double definiation of radio-meter noise!\n");
		exit (1);
	}
	printf ("Nchan: %d; Tsys: %lf; Tsky: %lf; Gain: %lf; Radio-meter noise: %lf mJy\n", control->nchan, control->tsys, control->tsky, control->gain, control->radioNoise);

  return finished;
}

void initialiseControl(controlStruct *control)
{
  strcpy(control->primaryHeaderParams,"UNKNOWN");
  strcpy(control->exact_ephemeris,"UNKNOWN");
  strcpy(control->fname,"UNKNOWN");
  strcpy(control->src,"UNKNOWN");
  control->tsub = 0;
  
  // Standard defaults
  strcpy(control->type,"PSR");
  control->nbin = 128;
  control->nchan = 1024;
  control->npol = 1;
  control->nsub = 1;
  control->cFreq = 1400.0;
  control->obsBW = -256;
  control->segLength = 48000;
  control->nfreqcoeff = 16;
  control->ntimecoeff = 16;
  control->stt_imjd = 55000;
  control->stt_smjd = 5234.0;
  control->stt_offs = 0.1234;
  control->tsubRequested = 60;
  control->whiteLevel = 0;
  control->scint_ts  = 0.0;
  control->scint_freqbw = 0.0;
  control->tsys = 0.0;
  control->tsky = 0.0;
  control->gain = 0.0;

  control->cFlux = 0.0;
  control->si = 0.0;
  control->radioNoise = 0.0;
  // Note that the DM comes from the ephemeris
}

/*
double calculateScintScale(controlStruct *control,int chan,long double timeFromStart,int sub,long int *seed,float **scint,int *setScint)
{
  double simFlux=0,fc;
  double boxX,boxY0,dt,ratio,ratio2,rf,sdiff,f2;
  double f0,f1,dstep,scint_ts_f,scint_freqbw_f;
  double tint = control->tsub;
  int nx=16384;
  int k2,l,nc;
  static int xs=0;
  double scint_ts_f0,fa,fb;
  int fend;
  printf("In scint\n");
  if (*setScint == 0)
    {
      int nx=16384;
      //int ny=2048;
      int ny=1024;
      FILE *fin;
      float *flt;
      int i,j;

      flt = (float *)malloc(sizeof(float)*nx*2); // Read in real and imaginary parts of the e-field       
      *setScint=1;
      // Read the file
      if (!(fin = fopen("strong10w.spe","rb")))
	{
	  printf("Unable to read scintillation file: strong10w.spe\n");
	  exit(1);
	}
      // Read header information
      fread(flt,sizeof(float),nx*2,fin);  
      for (j=0;j<ny;j++)
	{  
	  fread(flt,sizeof(float),nx*2,fin);  
	  for (i=0;i<nx;i++)
	    {
	      scint[j][i] = pow(flt[2*i],2)+pow(flt[2*i+1],2);
	    }
	}
      fclose(fin);
      free(flt);
    }
  printf("Set up scint\n");
  f0 = control->cFreq + fabs(control->obsBW)/2.0; // Highest frequency
  //fc = f0 - fabs(control->obsBW/(double)control->nchan)*chan + fabs(control->obsBW/(double)control->nchan)*0.5;
  fc = f0 - fabs(control->obsBW/(double)control->nchan)*chan;
  printf("fc = %g\n",fc);
  // Simulation parameters
  rf = 20;
  sdiff = 4.3;  // sdiff = rf*0.215
  
  // Deal with scintillation
  scint_ts_f     = control->scint_ts*pow(fc/1440,1.2); // 1440 is fixed from the simulation
  //scint_freqbw_f = control->scint_freqbw*pow(fc/1440,-4.4); 
  scint_freqbw_f = control->scint_freqbw*pow(fc/1440,4.4); 
  ratio = pow(rf/sdiff,2);
  //	  ratio2 = fc*1e6/data->scint_freqbw; 
  ratio2 = fc*1e6/scint_freqbw_f;  // JUST PUT THIS IN
  f2 = pow(10,log10(ratio2/ratio)/(-3.4));
  printf("f2 %g\n",f2);
  
  //	  scint_ts_f0 = data->scint_ts*pow(1440/f2/1440,1.2); // This is crazy?? MUST FIX
  scint_ts_f0 = scint_ts_f*pow(1.0/f2,1.2); //*pow(1440/f2/1440,1.2); // This is crazy?? MUST FIX
  printf("scint_ts %g %g\n",scint_ts_f,scint_ts_f0);
  //	  dt = scint_ts_f0/sdiff;
  dt = scint_ts_f0/sdiff;
  boxX = tint/dt;
  //scint_ts_f     = control->scint_ts*pow(fc/1440,1.2); // 1440 is fixed from the simulation

  printf("boxX = %g %g %g\n",boxX,tint,dt);
  //		  printf("dt %g %g %g %g %g %g %g\n",dt,scint_ts_f0,sdiff,f2,ratio2,ratio,data->scint_freqbw); 
  // Now find correct section in the simulation for the particular receiver band
  fa = f2*(fc-fabs(control->obsBW/(double)control->nchan))/1440.0; // Measurements made at 1.4GHz
  fb = f2*(fc)/1440;
  if (fa > 1.6) fa = 1.6; // Limit of simulation
  if (fa < 0.4) 
    {
      fb = 0.4+(fb-fa);
      fa = 0.4;
    }
  
  if (fb > 1.6) fb = 1.6; // Limit of simulation
  dstep = (1.6-0.4)/1024.0;
  if (sub==0 && chan==0)
    {
      xs = (int)(TKranDev(seed)*(nx)); //-3*86400.0/dt)); // Must fix
      printf("START Steps = %d %d %g\n",xs,nx,3*86400.0/dt);
    }
  else if (chan==0)
    xs = (int)(xs+control->tsub/dt+0.5);
  printf("timeFromStart = %d %Lg\n",xs,timeFromStart);
  //  printf("xs = %d dxs = %d fa = %g fb = %g dstep = %g %g\n",xs,(int)((1-1)*control->tsub/dt+0.5),fa,fb,dstep,(fb-0.4)/dstep);

  nc=0;
  simFlux=0;
  fend = (int)((fb-0.4)/dstep+1e-6); // There's a rounding issue here. Therefore increase the value slightly
  if (fend >= 1024) {printf("WARNING: Edge of scintillation matrix\n"); fend=1023;}
  if (chan==0)
    printf("Steps %g %g %d %d %d %d (%d)\n",(double)timeFromStart,dt,(int)xs,(int)(xs+boxX),(int)((fa-0.4)/dstep),fend,(int)boxX);
  if (boxX < 1) boxX = 1;
  for (k2=(int)xs;k2<(int)(xs+boxX);k2++)
    {
      
      for (l=(int)((fa-0.4)/dstep);l<=fend;l++)
	{
	  //	  printf("Calculation %.10f %.5f\n",(double)dstep,(double)(fb-0.4)/(dstep));
	  //	  printf("Processing %d %d %.5f >%g< %.5f %g\n",l,k2,fb,dstep,(double)(((long double)fb-0.4L)/(long double)dstep),boxX);
	  simFlux+=scint[l][k2]; // Should think carefully about this averaging
	  nc++;
	  //	  printf("Done process: %d %g %g\n",nc,scint[l][k2],simFlux);
	  
	}
    }
  if (nc == 0)
    {
      printf("SHOULD NEVER GET HERE %g %g %g %g %g %g\n",fa,fb,(fa-0.4)/dstep,(fb-0.4)/dstep,xs,boxX);
      exit(1);
    }
  
  simFlux/=(double)nc;
  
  printf("simFlux = %g %d\n",simFlux,nc);
  return simFlux;
}
*/
