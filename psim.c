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
  T2Predictor pred, predBat;
  channel *chan;
  controlStruct control;
  int nPhaseBins;
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
  double scintScale=1.0;
	acfStruct acfStructure; // dynamical spectrum

	double *phaseResolvedSI; 
  phaseSI *SIrotated;
  
  seed = TKsetSeed();

  initialiseTemplate(&tmpl);
  initialiseControl(&control);

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
   
		nPhaseBins = control.nbin;
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
      
		// predictor at SSB
		control.bat = 1;
    T2Predictor_Init(&predBat);
    runTempo2(&control);
    if (ret=T2Predictor_Read(&predBat,(char *)"t2pred.dat"))
		{
			printf("Error: unable to read predictor\n");
			exit(1);
		}

		// calculate the dynamic spectrum

		if (control.scint_ts > 0)
		{
			calculateScintScale (&acfStructure, &control);
		}

		// read phase-resolved spectral index
		phaseResolvedSI = (double *)malloc(sizeof(double)*nPhaseBins);
    SIrotated = (phaseSI *)malloc(sizeof(phaseSI)*control.nchan);
    for (i=0;i<control.nchan;i++)
		{
			SIrotated[i].nbin = control.nbin;
	  	SIrotated[i].val = (double *)malloc(sizeof(double)*control.nbin);
		}
      
		if (control.simProf != 0)
		{
			readSI(&control, phaseResolvedSI);
		}

		// calculate stt_offs
		//calculateStt_offs(&control, pred);

		//////////////////////////////////////////////////////////////
    timeFromStart = 0;
    for (i=0;i<control.nsub;i++)
		{
			// Calculate period for this subint
			calculatePeriod(&control,pred,timeFromStart);
			calculateBatPeriod(&control,predBat,timeFromStart);

			// Calculate phase offset for each channel
	  	for (j=0;j<control.nchan;j++)
	    {
				calculatePhaseOffset(j,&control,pred,timeFromStart);
			}

			if (control.simProf != 0)
			{
				for (j=0;j<control.nchan;j++)
				{
					rotateSI (phaseResolvedSI, nPhaseBins, control.phaseOffset[j], SIrotated[j].val);
				}
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
				else
				{
					scintScale = 1.0;
				}
	      
				tempFlux = 0.0;
				for (l=0;l<nPhaseBins;l++)
				{
					if (control.simProf == 0)
					{
						tempFlux += evaluateTemplate(&control,&tmpl,j,0,l);
					}
					else
					{
						tempFlux += simTemplate(&control,&tmpl,SIrotated[j].val[l],j,0,l);
					}
				}
				tempFlux = tempFlux/nPhaseBins;

				for (k=0;k<control.npol;k++)
				{
					for (l=0;l<nPhaseBins;l++)
					{
						if (control.simProf == 0)
						{
							//chan[j].pol[k].val[l] = control.whiteLevel*TKgaussDev(&seed);
							//chan[j].pol[k].val[l] += scintScale*evaluateTemplate(&control,&tmpl,j,k,l);
							//chan[j].pol[k].val[l] = scintScale*evaluateTemplate(&control,&tmpl,j,k,l)*control.flux[j]/tempFlux + control.radioNoise*TKgaussDev(&seed);
							chan[j].pol[k].val[l] = scintScale*evaluateTemplate(&control,&tmpl,j,k,l)*control.flux[j]/tempFlux + control.radioNoise*TKgaussDev(&seed);
							//printf ("%lf\n",evaluateTemplate(&control,&tmpl,j,k,l));
						}
						else
						{
							chan[j].pol[k].val[l] = scintScale*simTemplate(&control,&tmpl,SIrotated[j].val[l],j,k,l)*control.flux[j]/tempFlux + control.radioNoise*TKgaussDev(&seed);
						}
					}
				}
			}
	  
			writeChannels(chan,&control,fptr,i+1,timeFromStart);
			timeFromStart += control.tsub;
		}
		fits_close_file(fptr,&status);
    T2Predictor_Destroy(&pred);
    T2Predictor_Destroy(&predBat);

    // Deallocate memory
    free(control.phaseOffset);
    for (i=0;i<control.nchan;i++)
		{
			for (j=0;j<control.npol;j++)
				free(chan[i].pol[j].val);
			free(chan[i].pol);
		}
      
		free(chan);
		free(phaseResolvedSI);
		free(SIrotated);
	}
  
	deallocateMemory (&acfStructure);
	fclose(fin);
  // Should deallocate the memory
}

int readSI(controlStruct *control, double *phaseResolvedSI)
{
	FILE *fp;
	int i;
	double tmp;

	if ((fp = fopen(control->phaseResolvedSI,"r")) == NULL)
	{
		printf ("No phase-resolved spectral index!\n");
		exit(1);
	}

	i = 0;
	while (fscanf(fp, "%lf %lf %lf", &tmp, &phaseResolvedSI[i], &tmp) == 3)
	{
		i++;
	}

	if (fclose(fp) != 0)
		printf ("Can not close phase-resolved spectral index!\n");

	return 0;
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
  
	int ncol;
	fits_get_num_cols(fptr,&ncol,&status);

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

    fits_insert_col(fptr,ncol+1,"BATFREQ",(char *)"D",&status);
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
		double batFreq = control->batFreq;

		//printf ("DAI %.10lf\n", offsSub);
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

		//double CbatFreq;
		//if (subint == ceil(control->nsub/2))
		//{
		//	CbatFreq = control->CbatFreq;
		//	fits_write_key(fptr,TDOUBLE, (char *)"CBATFREQ",&CbatFreq,NULL,&status);  
		//}

    fits_get_colnum(fptr,CASEINSEN,"BATFREQ",&colnum,&status);
    fits_write_col(fptr,TDOUBLE,colnum,subint,1,1,&batFreq,&status);

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
			dat_freq[i] = f0-fabs(control->obsBW/(double)control->nchan)*i-fabs(control->obsBW/(double)control->nchan)*0.5; // Must fix
			//dat_freq[i] = f0-fabs(control->obsBW/(double)control->nchan)*i; // Must fix
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

	if (control->bat == 1)
	{
		sprintf(execString,"tempo2 -pred \"%s %Lf %Lf %g %g %d %d %g\" -f %s","BAT",mjd1,mjd2,freq1,freq2,ntimecoeff,nfreqcoeff,seg_length,control->exact_ephemeris);
	}
	else
	{
		sprintf(execString,"tempo2 -pred \"%s %Lf %Lf %g %g %d %d %g\" -f %s","PKS",mjd1,mjd2,freq1,freq2,ntimecoeff,nfreqcoeff,seg_length,control->exact_ephemeris);
	}
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

void calculateStt_offs(controlStruct *control, T2Predictor pred)
{
  long double mjd0;
  long double freq;
	long double phase0;

	freq = control->cFreq;
	mjd0 = control->stt_imjd + (control->stt_smjd)/86400.0L;
	control->period = 1.0/T2Predictor_GetFrequency(&pred,mjd0,freq);

  phase0 = T2Predictor_GetPhase(&pred,mjd0,freq);
  //control->stt_offs = (phase0 - floorl(phase0))*control->period;
  control->stt_offs = (ceill(phase0) - phase0)*control->period;
	//printf("Here with %g %g %g %g\n",(double)freq,(double)mjd0,(double)control->period,(double)control->tsub);
}

void calculateBatPeriod(controlStruct *control, T2Predictor pred, long double timeFromStart)
{
  long double mjd0;
  long double freq;
  long double toff;

  freq = control->cFreq;
  toff = (int)(control->tsub/2.0/control->period+0.5)*control->period;
  mjd0 = control->stt_imjd + (control->stt_smjd + control->stt_offs)/86400.0L + (timeFromStart + toff)/86400.0L; // Centre of subint
  control->batFreq = T2Predictor_GetFrequency(&pred,mjd0,freq);

	//if (nsub == control->nsub/2)
	//{
	//	control->CbatFreq = control->batFreq;
	//}
}

void calculatePeriod(controlStruct *control, T2Predictor pred, long double timeFromStart)
{
  long double mjd0;
  long double freq;
  long double toff;

  freq = control->cFreq;
  toff = (int)(control->tsubRequested/2.0/control->period+0.5)*control->period;
  mjd0 = control->stt_imjd + (control->stt_smjd + control->stt_offs)/86400.0L + (timeFromStart + toff)/86400.0L; // Centre of subint
  control->period = 1.0/T2Predictor_GetFrequency(&pred,mjd0,freq);
  control->tsub = ((int)(control->tsubRequested/control->period+0.5))*control->period;

  toff = (int)(control->tsub/2.0/control->period+0.5)*control->period;
  mjd0 = control->stt_imjd + (control->stt_smjd + control->stt_offs)/86400.0L + (timeFromStart + toff)/86400.0L; // Centre of subint
  control->period = 1.0/T2Predictor_GetFrequency(&pred,mjd0,freq);
  control->tsub = ((int)(control->tsubRequested/control->period+0.5))*control->period;

	/*
	// test
	long double mjd1;
	long double period1;
  mjd1 = control->stt_imjd + (control->stt_smjd + control->stt_offs)/86400.0L + (timeFromStart)/86400.0L; // Centre of subint
  period1 = 1.0/T2Predictor_GetFrequency(&pred,mjd1,freq);
	printf ("DAI %.10Lf %.10Lf\n", control->period, period1);
	//printf ("DAI %.15Lf\n", control->period-period1);
	
	if (sub == 0)
	{
		freq = control->cFreq;
		mjd0 = control->stt_imjd + (control->stt_smjd + control->stt_offs)/86400.0L + control->tsubRequested*0.5/86400.0L;
		control->period = 1.0/T2Predictor_GetFrequency(&pred,mjd0,freq);
		control->tsub = ((int)(control->tsubRequested/control->period+0.5))*control->period;

		//Iterate once more with tsub instead of tsubRequested
		mjd0 = control->stt_imjd + (control->stt_smjd + control->stt_offs)/86400.0L + control->tsub*0.5/86400.0L;
		control->period = 1.0/T2Predictor_GetFrequency(&pred,mjd0,freq);
		//control->tsub = ((int)(control->tsub/control->period+0.5))*control->period;
		control->tsub = ((int)(control->tsubRequested/control->period+0.5))*control->period;
	}
	else
	{
		freq = control->cFreq;
		//toff = 0.0;
		toff = (int)(control->tsub/2.0/control->period+0.5)*control->period;
		mjd0 = control->stt_imjd + (control->stt_smjd + control->stt_offs)/86400.0L + (timeFromStart + toff)/86400.0L;
		control->period = 1.0/T2Predictor_GetFrequency(&pred,mjd0,freq);
		//control->tsub = ((int)(control->tsub/control->period+0.5))*control->period;
		control->tsub = ((int)(control->tsubRequested/control->period+0.5))*control->period;

		//toff = 0.0;
		toff = (int)(control->tsub/2.0/control->period+0.5)*control->period;
		mjd0 = control->stt_imjd + (control->stt_smjd + control->stt_offs)/86400.0L + (timeFromStart + toff)/86400.0L;
		control->period = 1.0/T2Predictor_GetFrequency(&pred,mjd0,freq);
		//control->tsub = ((int)(control->tsub/control->period+0.5))*control->period;
		control->tsub = ((int)(control->tsubRequested/control->period+0.5))*control->period;
	} 
	*/
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
  	result += (tmpl->channel[chan].pol[stokes].comp[comp].vonMises[k].height) *
    	exp(tmpl->channel[chan].pol[stokes].comp[comp].vonMises[k].concentration*
	(cos((phi - tmpl->channel[chan].pol[stokes].comp[comp].vonMises[k].centroid + phiRot)*2*M_PI)-1));
  return result;
}

// Simulate a single template component
double simTemplateComponent(tmplStruct *tmpl, double freq, double SI, double phi,int chan,int stokes,int comp,double phiRot)
{
  double result=0;
  //result = tmpl->channel[chan].pol[stokes].comp[comp].height *
  int k;
  for (k=0;k<tmpl->channel[chan].pol[stokes].comp[comp].nVm;k++)
  	result += (tmpl->channel[chan].pol[stokes].comp[comp].vonMises[k].height)*pow(freq/1400.0,SI) *
    	exp(tmpl->channel[chan].pol[stokes].comp[comp].vonMises[k].concentration*
	(cos((phi - tmpl->channel[chan].pol[stokes].comp[comp].vonMises[k].centroid + phiRot)*2*M_PI)-1));
  return result;
}

// Simulate a given frequency channel and polarisation
double simTemplate(controlStruct *control, tmplStruct *tmpl, double SI, int chan, int pol, int bin)
{
  int tChan = (int)(chan*tmpl->nchan/control->nchan);

	double f0, freq;
  f0 = control->cFreq + fabs(control->obsBW)/2.0; // Highest frequency
  freq = f0 - fabs(control->obsBW/(double)control->nchan)*chan - fabs(control->obsBW/(double)control->nchan)*0.5;

  double result=0;
  int k;
  for (k=0;k<tmpl->channel[tChan].pol[pol].nComp;k++)
    {
      //result += evaluateTemplateComponent(tmpl,bin/(double)control->nbin,tChan,pol,k,0.0);
      result += simTemplateComponent(tmpl,freq,SI,bin/(double)control->nbin,tChan,pol,k,control->phaseOffset[chan]);
    }
  return result;
}

// Evaluate a given frequency channel and polarisation
double evaluateTemplate(controlStruct *control, tmplStruct *tmpl, int chan, int pol, int bin)
{
  int tChan = (int)(chan*tmpl->nchan/control->nchan);

  double result=0;
  int k;
  for (k=0;k<tmpl->channel[tChan].pol[pol].nComp;k++)
    {
      //result += evaluateTemplateComponent(tmpl,bin/(double)control->nbin,tChan,pol,k,0.0);
      result += evaluateTemplateComponent(tmpl,bin/(double)control->nbin,tChan,pol,k,control->phaseOffset[chan]);
    }
  return result;
}

void calculatePhaseOffset(int chan,controlStruct *control,T2Predictor pred,long double timeFromStart)
{
  long double f0,freq,phase0,mjd0;
  long double toff;

	////////////////////////////////////////////////////////////////////
	// Need to understand this
  toff = (int)(control->tsub/2.0/control->period+0.5)*control->period;
	//toff = 0.0;
	////////////////////////////////////////////////////////////////////
	
  mjd0 = control->stt_imjd + (control->stt_smjd + control->stt_offs)/86400.0L + (timeFromStart + toff)/86400.0L;
  f0 = control->cFreq + fabs(control->obsBW)/2.0; // Highest frequency
  freq = f0 - fabs(control->obsBW/(double)control->nchan)*chan - fabs(control->obsBW/(double)control->nchan)*0.5;
  //freq = f0 - fabs(control->obsBW/(double)control->nchan)*chan;
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
			else if (strcasecmp(param,"phaseResolvedSI")==0)
			{
				fscanf(fin,"%s",control->phaseResolvedSI);
				control->simProf = 1;
			}
		}
	} while (endit==0);

	control->bat = 0; // default
	control->CbatFreq = 0.0; // default
	//////////////////////////////////////////////////////////////////////
  f0 = control->cFreq + fabs(control->obsBW)/2.0; // Highest frequency
	control->flux = (double *)malloc(sizeof(double)*control->nchan);
	if (control->cFlux != 0.0 && control->si != 0.0)
	{
		for (i = 0; i < control->nchan; i++)
		{
			f1 = f0 - fabs(control->obsBW/(double)control->nchan)*i;
			control->flux[i] = pow(f1/1369.0, control->si)*control->cFlux;
		}
	}
	else 
	{
		printf ("cFlux and spectral index not provided!\n");
		for (i = 0; i < control->nchan; i++)
		{
			control->flux[i] = control->cFlux;
		}
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
	
	control->bat = 0;
	
	control->simProf = 0; // default: do not simulate profile with phase-resolved SI
  // Note that the DM comes from the ephemeris
}

int dft_profiles (int N, double *in, fftw_complex *out)
// dft of profiles
{
	//  dft of profiles 
	///////////////////////////////////////////////////////////////////////
	
	//printf ("%lf\n", in[0]);
	//double *in;
	//fftw_complex *out;
	fftw_plan p;
	
	//in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
	//out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
	p = fftw_plan_dft_r2c_1d(N, in, out, FFTW_MEASURE);

	fftw_execute(p);

	fftw_destroy_plan(p);
	//fftw_free(in); 
	//fftw_free(out);
  
	return 0;
}

int rotate (int N, double *real_p, double *real_p_rotate, double *ima_p, double *ima_p_rotate, double rot)
{
	// k is the dimention of amp, N is the dimention of s
	int i;

	// for substraction 
	double amp,cosina,sina;
	for (i=0;i<N/2+1;i++)
	{
		// calculate the sin(phi) and cos(phi) of the profile
		amp=sqrt(real_p[i]*real_p[i]+ima_p[i]*ima_p[i]);
		cosina=real_p[i]/amp;
		sina=ima_p[i]/amp;

		// rotate profile
		real_p_rotate[i]=amp*(cosina*cos(-i*rot*2*M_PI)-sina*sin(-i*rot*2*M_PI));
		ima_p_rotate[i]=amp*(sina*cos(-i*rot*2*M_PI)+cosina*sin(-i*rot*2*M_PI));
		//real_p_rotate[i]=amp*(cosina*cos(-i*pi)-sina*sin(-i*pi));
		//ima_p_rotate[i]=amp*(sina*cos(-i*pi)+cosina*sin(-i*pi));
		
	}

	return 0;
}

int inverse_dft (double *real_p, double *ima_p, int ncount, double *p_new)
{
	double *dp;
	fftw_plan plan;
	fftw_complex *cp;

	dp = (double *)malloc(sizeof (double) * ncount);
	cp = (fftw_complex *)fftw_malloc(sizeof (fftw_complex) * ncount);
	memset(dp, 0, sizeof (double) * ncount);
	memset(cp, 0, sizeof (fftw_complex) * ncount);

	// initialize the dft...
	double *dp_t;
	fftw_plan plan_t;
	fftw_complex *cp_t;

	dp_t = (double *)malloc(sizeof (double) * ncount);
	cp_t = (fftw_complex *)fftw_malloc(sizeof (fftw_complex) * ncount);
	memset(dp_t, 0, sizeof (double) * ncount);
	memset(cp_t, 0, sizeof (fftw_complex) * ncount);

	int i;
	double real,ima,amp,cosina,sina;

	for (i = 0; i < ncount; i++)
	{
		if (i < ncount/2+1)
		{
			real = real_p[i];
			ima = ima_p[i];
			amp = sqrt(real*real+ima*ima);
			cosina = real/amp;
			sina = ima/amp;

			cp[i][0] = amp*(cosina);
			cp[i][1] = amp*(sina);
			//cp[i][0] = amp*(cosina*cos(-i*3.1415926)-sina*sin(-i*3.1415926));
			//cp[i][1] = amp*(sina*cos(-i*3.1415926)+cosina*sin(-i*3.1415926));
			//cp[i][0]=real_s[i]-real_p[i];
			//cp[i][1]=ima_s[i]-ima_p[i];
			//cp[i][0]=-real_s[i]+real_p[i];
			//cp[i][1]=-ima_s[i]+ima_p[i];
			cp_t[i][0] = real_p[i];
			cp_t[i][1] = ima_p[i];
			//cp[i][0]=real_p[i];
			//cp[i][1]=ima_p[i];
		}
		else
		{
			cp[i][0]=0.0;
			cp[i][1]=0.0;
			cp_t[i][0]=0.0;
			cp_t[i][1]=0.0;
		}
	}

  plan_t = fftw_plan_dft_c2r_1d(ncount, cp_t, dp_t, FFTW_MEASURE);

  fftw_execute(plan_t);

  fftw_destroy_plan(plan_t);

	///////////////////////////////////////////////////////////////

  plan = fftw_plan_dft_c2r_1d(ncount, cp, dp, FFTW_MEASURE);

  fftw_execute(plan);

  fftw_destroy_plan(plan);

	for (i = 0; i < ncount; i++)
	{
		p_new[i] = dp[i]/ncount;  // normalized by the ncount
		//printf ("%lf\n", p_new[i]);
	}

	return 0;
}

int rotateSI (double *s, int nphase, double rot, double *sOut)
{
	//int nphase=1024;
	int nchn=1;

	// dft 
	double s_real[nphase], s_ima[nphase];
	preRot (s, nphase, nchn, s_real, s_ima);

	// rotate the profile by pi
	double real_s_rotate[nphase/2+1], ima_s_rotate[nphase/2+1];
	rotate (nphase, s_real, real_s_rotate, s_ima, ima_s_rotate, rot);

	inverse_dft (real_s_rotate, ima_s_rotate, nphase, sOut);

	//printf ("%.8lf %.8lf %.8lf %.8lf %.8lf\n", sigma_I, sigma_Q, sigma_U, sigma_V, sigma_L);

	return 0;
}

int preRot (double *p, int nphase, int nchn, double *real_p, double *ima_p)
{
	// nphase is the dimention of one profile, nchn is number of profiles
	int i,j;
	
	/////////////////////////////////////////////////////////////////////////////////
	double test[nphase];  

	for (i=0;i<nphase;i++)
	{
		test[i]=p[i];
	}
	fftw_complex *out_t;
	out_t = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nphase);
	dft_profiles(nphase,test,out_t);
	//////////////////////////////////////////////////////////////////////////////

	fftw_complex *out_p;
	
	out_p = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nphase);
	
	double p_temp[nphase];  // store one template and profile

	for (i = 0; i < nchn; i++)
	{
	    for (j=0;j<nphase;j++)
	    {
		    p_temp[j]=p[i*nphase + j];
	    }

	    dft_profiles(nphase,p_temp,out_p);

	    //double amp_s[N/2],phi_s[N/2];
	    //double amp_p[N/2],phi_p[N/2];

		for (j = 0; j < nphase/2+1; j++)                                                  
		{                                                                      
			real_p[j]=out_p[j][0];                                             
			ima_p[j]=out_p[j][1];                                              
		}
										
	}

	fftw_free(out_p); 
	fftw_free(out_t); 

	return 0;
}

