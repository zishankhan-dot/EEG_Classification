# EEG_Classification
Analysis Application Using the PhysioNet EEG Motor Movement/Imagery Dataset

Instruction to Run Create Model locally :

1) download files (EEG PhysioNet runs-> [4,8,12]):
    if (linux){
    /scripts -> it have download_files.sh (linux bash script )which would download files for given range (change 66->starting index you want and 90-> end index)}

    else{
    download the files and save it in the directory ./data/raw as this have been used throughout the program for local development of models }

2) file(./src/model_pipeline/trainmodel.py):
    this file will automatically train model 
    pipeline followed :
    1)For each subjects join all the run then stack all subject together to create one big data 
    2)Filter the data out using alpha and beta filter (8-30)
    3)Create epochs (tmin=0, tmax=4) for T1 (left hand) and T2 (right hand) events only
    4)Extract features using Welch PSD, compute log band power for mu (8-13 Hz) and beta (13-30 Hz) bands across C3, C4, Cz channels giving 6 features per epoch
    5)Train LDA, SVM, RF, MLP models using StratifiedKFold cross validation and save each as a .pkl bundle in ./models/

3) to run the streamlit app locally :
    streamlit run app.py
    app will load in browser, either upload a PhysioNet .edf file (runs 4, 8, or 12) or use the built in demo subject (S029 R04) to get started

note : only subjects with classification accuracy above 65 percent were used for training, rest were excluded due to noise in the signal, tried various approaches to resolve this but did not work out

Read full summary report in ProgressReport.md




