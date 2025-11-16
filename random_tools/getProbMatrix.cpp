#include <cstdlib>
#include <vector>
#include <iostream>
#include <fstream>
#include <map>
#include <string>
#include <deque>

#include "TFile.h"
#include "TTree.h"
#include "TString.h"
#include "TSystem.h"
#include "TROOT.h"
#include "TMath.h"
#include <TChain.h>

using namespace std;

void getProbMatrix()
{
    TChain *c = new TChain("HESingles");
    for(Int_t iter = 1; iter <= 1000; iter++)
    {
        TString fileName;
//         fileName.Form("/media/sf_shared/J_Modular/ImageQuality/output/results1.root");
        fileName.Form("/mnt/home/sparzych/workdir/J_Modular/ImageQuality/output/results%i.root",iter);
        c->Add(fileName);
    }
    
    c->SetBranchStatus("*",0);
    c->SetBranchStatus("time",1);
    c->SetBranchStatus("rsectorID",1);
    c->SetBranchStatus("crystalID",1);
    c->SetBranchStatus("globalPosX",1);
    c->SetBranchStatus("globalPosY",1);
    c->SetBranchStatus("globalPosZ",1);
    
    Double_t time;
    Int_t rsectorID, crystalID;
    Float_t globalPosX, globalPosY, globalPosZ;
    
    c->SetBranchAddress("time",&time);
    c->SetBranchAddress("rsectorID",&rsectorID);
    c->SetBranchAddress("crystalID",&crystalID);
    c->SetBranchAddress("globalPosX",&globalPosX);
    c->SetBranchAddress("globalPosY",&globalPosY);
    c->SetBranchAddress("globalPosZ",&globalPosZ);
    
    Double_t shift = 200. * pow(10.,-9.);
    Double_t window = 3. * pow(10.,-9.);
    
    float prob[1200][1200] = {};
    
    Double_t dtwNumber = 0;
    
    std::deque <Double_t> t;
    std::deque <Int_t> R;
    std::deque <Int_t> cr;
    std::deque <Int_t> used;
    std::deque <Float_t> x;
    std::deque <Float_t> y;
    std::deque <Float_t> z;
    
    long int dequeEnd = 0;
    
    for(long int i = 0; i < c->GetEntries(); i++)
    {
        
        for(long int k = dequeEnd; k < c->GetEntries(); k++)
        {
            c->GetEntry(k);
            t.push_back(time);
            R.push_back(rsectorID);
            cr.push_back(crystalID);
            x.push_back(globalPosX);
            y.push_back(globalPosY);
            z.push_back(globalPosZ);
            used.push_back(0);
            dequeEnd++;
            if(time > t[0] + shift + window){break;}
        }
        
        Double_t primeTime = t[0];
        Int_t primeRsector = R[0];
        Int_t primeCrystal = cr[0];
        Double_t secondTime = -666.;
        Int_t secondRsector = -666;
        Int_t secondCrystal = -666;
        long int usedEvent = -666;
        Double_t globalPosX1 = x[0];
        Double_t globalPosY1 = y[0];
        Double_t globalPosZ1 = z[0];
        Double_t globalPosX2 = -666.;
        Double_t globalPosY2 = -666.;
        Double_t globalPosZ2 = -666.;
        
        Int_t coincidenceFlag = 0;
        
        if(used[0] == 0)    //Check if the 'first' photon wasn't already used
        {
            for(long int j = 1; j < t.size(); j++)
            {
                if(t[j] < primeTime + shift){continue;}
                else if(t[j] > primeTime + shift + window){break;}
                else if(used[j] == 0 && R[j] != primeRsector)   //Check if the 'second' photon wasn't already used
                {
                    if(coincidenceFlag == 0)
                    {
                        coincidenceFlag++;
                        secondTime = t[j];
                        secondRsector = R[j];
                        secondCrystal = cr[j];
                        globalPosX2 = x[j];
                        globalPosY2 = y[j];
                        globalPosZ2 = z[j];
                        usedEvent = j;
                    }
                    else
                    {
                        coincidenceFlag = -666;
                        break;
                    }
                }
            }
        }
        
        if(coincidenceFlag == 1)
        {
            dtwNumber++;
            used[usedEvent] = 1;
            Double_t dt = abs(primeTime*pow(10.,9) - secondTime*pow(10.,9));
            Float_t angle = globalPosX1*globalPosX2 + globalPosY1*globalPosY2 + globalPosZ1*globalPosZ2;
            angle /= sqrt( globalPosX1*globalPosX1 + globalPosY1*globalPosY1 + globalPosZ1*globalPosZ1 );
            angle /= sqrt( globalPosX2*globalPosX2 + globalPosY2*globalPosY2 + globalPosZ2*globalPosZ2 );
            angle = acos( angle ) * 180. / TMath::Pi();
            Int_t rsector1 = primeRsector;
            Int_t rsector2 = secondRsector;
            Int_t crystal1 = primeCrystal;
            Int_t crystal2 = secondCrystal;
            Float_t posZ1 = globalPosZ1;
            Float_t posZ2 = globalPosZ2;
            Double_t z1 = posZ1 + 500./2.;
            Double_t z2 = posZ2 + 500./2.;
            Int_t i1 = ((Int_t)rsector1*50) + (Int_t)(z1/10);
            Int_t i2 = ((Int_t)rsector2*50) + (Int_t)(z2/10);
            prob[i1][i2]++;
            prob[i2][i1]++;
        }
        
        t.pop_front();
        R.pop_front();
        cr.pop_front();
        x.pop_front();
        y.pop_front();
        z.pop_front();
        used.pop_front();
    }
    
    t.clear();
    R.clear();
    cr.clear();
    x.clear();
    y.clear();
    z.clear();
    used.clear();
    
//--------------------------------------------------------
    
    TChain *MCtree = new TChain("Coincidences");
    for(Int_t iter = 1; iter <= 1000; iter++)
    {
        TString fileName2;
//         fileName2.Form("/media/sf_shared/J_Modular/ImageQuality/output/results1.root");
        fileName2.Form("/mnt/home/sparzych/workdir/J_Modular/ImageQuality/output/results%i.root",iter);
        MCtree->Add(fileName2);
    }
    
    MCtree->SetBranchStatus("*",0);
    MCtree->SetBranchStatus("eventID1",1);
    MCtree->SetBranchStatus("time1",1);
    MCtree->SetBranchStatus("globalPosX1",1);
    MCtree->SetBranchStatus("globalPosY1",1);
    MCtree->SetBranchStatus("globalPosZ1",1);
    MCtree->SetBranchStatus("rsectorID1",1);
    MCtree->SetBranchStatus("comptonPhantom1",1);
    MCtree->SetBranchStatus("comptonCrystal1",1);
    MCtree->SetBranchStatus("RayleighPhantom1",1);
    MCtree->SetBranchStatus("eventID2",1);
    MCtree->SetBranchStatus("time2",1);
    MCtree->SetBranchStatus("globalPosX2",1);
    MCtree->SetBranchStatus("globalPosY2",1);
    MCtree->SetBranchStatus("globalPosZ2",1);
    MCtree->SetBranchStatus("rsectorID2",1);
    MCtree->SetBranchStatus("comptonPhantom2",1);
    MCtree->SetBranchStatus("comptonCrystal2",1);
    MCtree->SetBranchStatus("RayleighPhantom2",1);
    
    Int_t eventID1;
    Double_t time1;
    Float_t globalPosX1;
    Float_t globalPosY1;
    Float_t globalPosZ1;
    Int_t rsectorID1;
    Int_t comptonPhantom1;
    Int_t comptonCrystal1;
    Int_t RayleighPhantom1;
    Int_t eventID2;
    Double_t time2;
    Float_t globalPosX2;
    Float_t globalPosY2;
    Float_t globalPosZ2;
    Int_t rsectorID2;
    Int_t comptonPhantom2;
    Int_t comptonCrystal2;
    Int_t RayleighPhantom2;
    
    MCtree->SetBranchAddress("eventID1",&eventID1);
    MCtree->SetBranchAddress("time1",&time1);
    MCtree->SetBranchAddress("globalPosX1",&globalPosX1);
    MCtree->SetBranchAddress("globalPosY1",&globalPosY1);
    MCtree->SetBranchAddress("globalPosZ1",&globalPosZ1);
    MCtree->SetBranchAddress("rsectorID1",&rsectorID1);
    MCtree->SetBranchAddress("comptonPhantom1",&comptonPhantom1);
    MCtree->SetBranchAddress("comptonCrystal1",&comptonCrystal1);
    MCtree->SetBranchAddress("RayleighPhantom1",&RayleighPhantom1);
    MCtree->SetBranchAddress("eventID2",&eventID2);
    MCtree->SetBranchAddress("time2",&time2);
    MCtree->SetBranchAddress("globalPosX2",&globalPosX2);
    MCtree->SetBranchAddress("globalPosY2",&globalPosY2);
    MCtree->SetBranchAddress("globalPosZ2",&globalPosZ2);
    MCtree->SetBranchAddress("rsectorID2",&rsectorID2);
    MCtree->SetBranchAddress("comptonPhantom2",&comptonPhantom2);
    MCtree->SetBranchAddress("comptonCrystal2",&comptonCrystal2);
    MCtree->SetBranchAddress("RayleighPhantom2",&RayleighPhantom2);
    
    float MC[1200][1200] = {};
    
    for(long int i = 0; i < MCtree->GetEntries(); i++)
    {
        MCtree->GetEntry(i);
        if( (eventID1 != eventID2) || (eventID1 == eventID2 && comptonPhantom1 == 0 && comptonPhantom2 == 0 && comptonCrystal1 == 1 && comptonCrystal2 == 1 && RayleighPhantom1 == 0 && RayleighPhantom2 == 0) )
        {
            Double_t dt = abs(time1*pow(10.,9)-time2*pow(10.,9));
            Float_t angle = globalPosX1*globalPosX2+globalPosY1*globalPosY2+globalPosZ1*globalPosZ2;
            angle /= sqrt(globalPosX1*globalPosX1+globalPosY1*globalPosY1+globalPosZ1*globalPosZ1);
            angle /= sqrt(globalPosX2*globalPosX2+globalPosY2*globalPosY2+globalPosZ2*globalPosZ2);
            angle = acos(angle)*180./TMath::Pi();
            
            Float_t z1 = globalPosZ1 + 500./2.;
            Float_t z2 = globalPosZ2 + 500./2.;
            Int_t i1 = (rsectorID1*50) + (Int_t)(z1/10);
            Int_t i2 = (rsectorID2*50) + (Int_t)(z2/10);
            MC[i1][i2]++;
            MC[i2][i1]++;
        }
    }
    
//--------------------------------------------------------
    
    for(Int_t i = 0; i < 1200; i++)
    {
        for(Int_t j = 0; j < 1200; j++)
        {
            Float_t N = MC[i][j];
            if(N==0)
                prob[i][j] = 0.;
            else
            {
                Float_t R = prob[i][j];
                Float_t fraction = (N-R)/N;
                if(fraction < 0)
                    prob[i][j] = 0.;
                else
                    prob[i][j] = fraction;
            }
        }
    }
    
    ofstream output;
    output.open("probMatrix.txt");
    
    for(Int_t i = 0; i < 1200; i++)
    {
        for(Int_t j = 0; j < 1200; j++)
        {
            output << prob[i][j] << " ";
        }
        output << endl;
    }
    
    output.close();
    
    cout << std::setprecision(20) << "DTW = " << dtwNumber << endl;
    
}
