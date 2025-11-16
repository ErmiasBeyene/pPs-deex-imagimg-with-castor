#include <cstdlib>
#include <vector>
#include <iostream>
#include <map>
#include <string>
#include <fstream>
#include <TMath.h>
#include <TRandom.h>
#include <TString.h>
#include <TPRegexp.h>
#include <TChain.h>
#include <TH2F.h>

#include "TFile.h"
#include "TTree.h"
#include "TString.h"
#include "TSystem.h"
#include "TROOT.h"

using namespace std;


void corr_Tree(const std::string& probMatrixFile, const std::string&  inFile, const std::string& outFile)
{
    
//     -----------------------------------------------------------------------------------------------------------
    
    TChain *MCtree = new TChain("Coincidences");
    MCtree->Add(inFile.c_str());
    
    Int_t runID;
    Float_t axialPos;
    Float_t rotationAngle;
    Int_t eventID1;
    Int_t sourceID1;
    Float_t sourcePosX1;
    Float_t sourcePosY1;
    Float_t sourcePosZ1;
    Double_t time1;
    Float_t energy1;
    Float_t globalPosX1;
    Float_t globalPosY1;
    Float_t globalPosZ1;
    Int_t gantryID1;
    Int_t rsectorID1;
    Int_t moduleID1;
    Int_t submoduleID1;
    Int_t crystalID1;
    Int_t layerID1;
    Int_t comptonPhantom1;
    Int_t comptonCrystal1;
    Int_t RayleighPhantom1;
    Int_t RayleighCrystal1;
    Int_t eventID2;
    Int_t sourceID2;
    Float_t sourcePosX2;
    Float_t sourcePosY2;
    Float_t sourcePosZ2;
    Double_t time2;
    Float_t energy2;
    Float_t globalPosX2;
    Float_t globalPosY2;
    Float_t globalPosZ2;
    Int_t gantryID2;
    Int_t rsectorID2;
    Int_t moduleID2;
    Int_t submoduleID2;
    Int_t crystalID2;
    Int_t layerID2;
    Int_t comptonPhantom2;
    Int_t comptonCrystal2;
    Int_t RayleighPhantom2;
    Int_t RayleighCrystal2;
    Float_t sinogramTheta;
    Float_t sinogramS;
    Char_t comptVolName1[40];
    Char_t comptVolName2[40];
    Char_t RayleighVolName1[40];
    Char_t RayleighVolName2[40];
    
    MCtree->SetBranchAddress("runID",&runID);
    MCtree->SetBranchAddress("axialPos",&axialPos);
    MCtree->SetBranchAddress("rotationAngle",&rotationAngle);
    MCtree->SetBranchAddress("eventID1",&eventID1);
    MCtree->SetBranchAddress("sourceID1",&sourceID1);
    MCtree->SetBranchAddress("sourcePosX1",&sourcePosX1);
    MCtree->SetBranchAddress("sourcePosY1",&sourcePosY1);
    MCtree->SetBranchAddress("sourcePosZ1",&sourcePosZ1);
    MCtree->SetBranchAddress("time1",&time1);
    MCtree->SetBranchAddress("energy1",&energy1);
    MCtree->SetBranchAddress("globalPosX1",&globalPosX1);
    MCtree->SetBranchAddress("globalPosY1",&globalPosY1);
    MCtree->SetBranchAddress("globalPosZ1",&globalPosZ1);
    MCtree->SetBranchAddress("gantryID1",&gantryID1);
    MCtree->SetBranchAddress("rsectorID1",&rsectorID1);
    MCtree->SetBranchAddress("moduleID1",&moduleID1);
    MCtree->SetBranchAddress("submoduleID1",&submoduleID1);
    MCtree->SetBranchAddress("crystalID1",&crystalID1);
    MCtree->SetBranchAddress("layerID1",&layerID1);
    MCtree->SetBranchAddress("comptonPhantom1",&comptonPhantom1);
    MCtree->SetBranchAddress("comptonCrystal1",&comptonCrystal1);
    MCtree->SetBranchAddress("RayleighPhantom1",&RayleighPhantom1);
    MCtree->SetBranchAddress("RayleighCrystal1",&RayleighCrystal1);
    MCtree->SetBranchAddress("eventID2",&eventID2);
    MCtree->SetBranchAddress("sourceID2",&sourceID2);
    MCtree->SetBranchAddress("sourcePosX2",&sourcePosX2);
    MCtree->SetBranchAddress("sourcePosY2",&sourcePosY2);
    MCtree->SetBranchAddress("sourcePosZ2",&sourcePosZ2);
    MCtree->SetBranchAddress("time2",&time2);
    MCtree->SetBranchAddress("energy2",&energy2);
    MCtree->SetBranchAddress("globalPosX2",&globalPosX2);
    MCtree->SetBranchAddress("globalPosY2",&globalPosY2);
    MCtree->SetBranchAddress("globalPosZ2",&globalPosZ2);
    MCtree->SetBranchAddress("gantryID2",&gantryID2);
    MCtree->SetBranchAddress("rsectorID2",&rsectorID2);
    MCtree->SetBranchAddress("moduleID2",&moduleID2);
    MCtree->SetBranchAddress("submoduleID2",&submoduleID2);
    MCtree->SetBranchAddress("crystalID2",&crystalID2);
    MCtree->SetBranchAddress("layerID2",&layerID2);
    MCtree->SetBranchAddress("comptonPhantom2",&comptonPhantom2);
    MCtree->SetBranchAddress("comptonCrystal2",&comptonCrystal2);
    MCtree->SetBranchAddress("RayleighPhantom2",&RayleighPhantom2);
    MCtree->SetBranchAddress("RayleighCrystal2",&RayleighCrystal2);
    MCtree->SetBranchAddress("sinogramTheta",&sinogramTheta);
    MCtree->SetBranchAddress("sinogramS",&sinogramS);
    MCtree->SetBranchAddress("comptVolName1",comptVolName1);
    MCtree->SetBranchAddress("comptVolName2",comptVolName2);
    MCtree->SetBranchAddress("RayleighVolName1",RayleighVolName1);
    MCtree->SetBranchAddress("RayleighVolName2",RayleighVolName2);
    
//     -----------------------------------------------------------------------------------------------------------
    
    float prob[1200][1200] = {};
    
    ifstream input;
    input.open(probMatrixFile.c_str());
    
    for(int i = 0; i < 1200; i++)
    {
        for(int j = 0; j < 1200; j++)
        {
            float fraction;
            input >> fraction;
            prob[i][j] = fraction;
        }
    }
    
    input.close();
    
//     -----------------------------------------------------------------------------------------------------------
    
    TFile *file = new TFile(outFile.c_str(),"RECREATE");
    TTree *outTree = new TTree("Coincidences","Coincidences");
    
    outTree->Branch("runID",&runID);
    outTree->Branch("axialPos",&axialPos);
    outTree->Branch("rotationAngle",&rotationAngle);
    outTree->Branch("eventID1",&eventID1);
    outTree->Branch("sourceID1",&sourceID1);
    outTree->Branch("sourcePosX1",&sourcePosX1);
    outTree->Branch("sourcePosY1",&sourcePosY1);
    outTree->Branch("sourcePosZ1",&sourcePosZ1);
    outTree->Branch("time1",&time1);
    outTree->Branch("energy1",&energy1);
    outTree->Branch("globalPosX1",&globalPosX1);
    outTree->Branch("globalPosY1",&globalPosY1);
    outTree->Branch("globalPosZ1",&globalPosZ1);
    outTree->Branch("gantryID1",&gantryID1);
    outTree->Branch("rsectorID1",&rsectorID1);
    outTree->Branch("moduleID1",&moduleID1);
    outTree->Branch("submoduleID1",&submoduleID1);
    outTree->Branch("crystalID1",&crystalID1);
    outTree->Branch("layerID1",&layerID1);
    outTree->Branch("comptonPhantom1",&comptonPhantom1);
    outTree->Branch("comptonCrystal1",&comptonCrystal1);
    outTree->Branch("RayleighPhantom1",&RayleighPhantom1);
    outTree->Branch("RayleighCrystal1",&RayleighCrystal1);
    outTree->Branch("eventID2",&eventID2);
    outTree->Branch("sourceID2",&sourceID2);
    outTree->Branch("sourcePosX2",&sourcePosX2);
    outTree->Branch("sourcePosY2",&sourcePosY2);
    outTree->Branch("sourcePosZ2",&sourcePosZ2);
    outTree->Branch("time2",&time2);
    outTree->Branch("energy2",&energy2);
    outTree->Branch("globalPosX2",&globalPosX2);
    outTree->Branch("globalPosY2",&globalPosY2);
    outTree->Branch("globalPosZ2",&globalPosZ2);
    outTree->Branch("gantryID2",&gantryID2);
    outTree->Branch("rsectorID2",&rsectorID2);
    outTree->Branch("moduleID2",&moduleID2);
    outTree->Branch("submoduleID2",&submoduleID2);
    outTree->Branch("crystalID2",&crystalID2);
    outTree->Branch("layerID2",&layerID2);
    outTree->Branch("comptonPhantom2",&comptonPhantom2);
    outTree->Branch("comptonCrystal2",&comptonCrystal2);
    outTree->Branch("RayleighPhantom2",&RayleighPhantom2);
    outTree->Branch("RayleighCrystal2",&RayleighCrystal2);
    outTree->Branch("sinogramTheta",&sinogramTheta);
    outTree->Branch("sinogramS",&sinogramS);
    outTree->Branch("comptVolName1",comptVolName1,"comptVolName1[40]/C");
    outTree->Branch("comptVolName2",comptVolName2,"comptVolName2[40]/C");
    outTree->Branch("RayleighVolName1",RayleighVolName1,"RayleighVolName1[40]/C");
    outTree->Branch("RayleighVolName2",RayleighVolName2,"RayleighVolName2[40]/C");
    
    for(long int i = 0; i < MCtree->GetEntries(); i++)
    {
        MCtree->GetEntry(i);
        if( (eventID1 != eventID2) || (eventID1 == eventID2 && comptonPhantom1 == 0 && comptonPhantom2 == 0 && /*comptonCrystal1 == 1 && comptonCrystal2 == 1 &&*/ RayleighPhantom1 == 0 && RayleighPhantom2 == 0) )
        {
            //Double_t dt = abs(time1*pow(10.,9)-time2*pow(10.,9));
            Float_t angle = globalPosX1*globalPosX2+globalPosY1*globalPosY2+globalPosZ1*globalPosZ2;
            angle /= sqrt(globalPosX1*globalPosX1+globalPosY1*globalPosY1+globalPosZ1*globalPosZ1);
            angle /= sqrt(globalPosX2*globalPosX2+globalPosY2*globalPosY2+globalPosZ2*globalPosZ2);
            angle = acos(angle)*180./TMath::Pi();
            
            Float_t z1 = globalPosZ1 + 500./2.;
            Float_t z2 = globalPosZ2 + 500./2.;
            Int_t i1 = (rsectorID1*50) + (Int_t)(z1/10);
            Int_t i2 = (rsectorID2*50) + (Int_t)(z2/10);
            
            float fraction = prob[i1][i2];
            if(gRandom->Uniform(0,1) <= fraction)
                outTree->Fill();
        }
            
    }
    
    outTree->Write();
    file->Close();
    
}

int main(int argc, char *argv[])
{
  std::string probMatrixFile = "probMatrix.txt";
  std::string inFile = "results.root";
  std::string outFile = "resultsFiltered.root";
  if (argc == 4) 
  {
    probMatrixFile = std::string(argv[1]);
    inFile = std::string(argv[2]);
    outFile = std::string(argv[3]);

  }
  corr_Tree(probMatrixFile, inFile, outFile);
  return 0;
}
