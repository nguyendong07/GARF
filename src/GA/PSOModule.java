/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package GA;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.Map.Entry;
import java.util.Collection;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomForest;

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;

public class PSOModule {

    protected static double fitness(int[] x) {
        System.out.println("bsP_BagSizePercent: " + x[0] + "          nI_NumIterations: " + x[1]);
        double fitness = 0;
        try {
            fitness = RandomForestDemo(x[0], x[1]);
        } catch (Exception ex) {
           // Logger.getLogger(PSOrandomForest.class.getName()).log(Level.SEVERE, null, ex);
        }
        System.out.println(fitness);
        System.out.println("--------------------------------------------------------");
        System.out.println("");
        return (fitness);

    }

    public static void main(String[] args) {
        double w = 0.9,
                c1 = 2.05, c2 = 2.05,
                r1 = 0.0, r2 = 0.0,
                //xMin      = -5.12, xMax      = 5.12,
                vMin = -10, vMax = 10,
                nInfinite = 0.5,
                gBestValue = 0.5;
//
        int Np = 7, // # of particles
            Nd = 2, // # of dimensions
            Nt = 100; // # of Time Steps

        double[] pBestValue = new double[Np],
                gBestPosition = new double[Nd],
                bestFitnessHistory = new double[Nt],
                M = new double[Np],
                xMin = new double[Nd],
                xMax = new double[Nd];
        xMin[0] = 50;
        xMax[0] = 85; //bsP_BagSizePercent
        xMin[1] = 50;
        xMax[1] = 1000; //nI_NumIterations
        double[][] pBestPosition = new double[Np][Nd],
                   V = new double[Np][Nd];
        int[][] R = new int[Np][Nd];
        Random rand = new Random();

        for (int p = 0; p < Np; p++) {
            pBestValue[p] = nInfinite;
        }

        for (int p = 0; p < Np; p++) {
            for (int i = 0; i < Nd; i++) {
                R[p][i] = (int) (xMin[i] + (xMax[i] - xMin[i]) * rand.nextDouble());
                V[p][i] = vMin + (vMax - vMin) * rand.nextDouble();
            }
        }

        for (int p = 0; p < Np; p++) {
            M[p] = 0.5;
        }

        for (int j = 0; j < Nt; j++) {
            for (int p = 0; p < Np; p++) {
                for (int i = 0; i < Nd; i++) {
                    R[p][i] = (int) (R[p][i] + V[p][i]);

                    if (R[p][i] > xMax[i]) {
                        R[p][i] = (int) xMax[i];
                    } else if (R[p][i] < xMin[i]) {
                        R[p][i] = (int) xMin[i];
                    }
                }
            }

            for (int p = 0; p < Np; p++) {
                M[p] = fitness(R[p]);
                if (M[p] > pBestValue[p]) {
                    pBestValue[p] = M[p];
                    for (int i = 0; i < Nd; i++) {
                        pBestPosition[p][i] = R[p][i];
                    }
                }

                if (M[p] > gBestValue) {
                    gBestValue = M[p];
                    for (int i = 0; i < Nd; i++) {
                        gBestPosition[i] = R[p][i];
                    }
                }

            }
            bestFitnessHistory[j] = gBestValue;

            for (int p = 0; p < Np; p++) {
                for (int i = 0; i < Nd; i++) {
                    r1 = rand.nextDouble();
                    r2 = rand.nextDouble();
                    V[p][i] = V[p][i] + r1 * c1 * (pBestPosition[p][i] - R[p][i]) + r2 * c2 * (gBestPosition[i] - R[p][i]);
                    // classic Velocity update formulate                
                    if (V[p][i] > vMax) {
                        V[p][i] = vMax;
                    } else if (V[p][i] < vMin) {
                        V[p][i] = vMin;
                    }
                }
            }
            //output global best value at current timestep
            System.out.println("iteration: " + j + " BestValue " + gBestValue);
            
            System.out.println("gBestPosition_BagSizePercent: " + gBestPosition[0] + " gBestPosition_NumIterations: " + gBestPosition[1]);
        }
    }

    private static double RandomForestDemo(int bsP_BagSizePercent, int nI_NumIterations) throws Exception {

        //convertCSV2AARFF("D:\\DataTest\\acc_0.5\\acc_2_0.5_model_0_Status.csv", "D:\\DataTest\\acc_0.5\\Test.arff");
        // load data
        BufferedReader br = null;
        br = new BufferedReader(new FileReader("C:\\Users\\dongnv\\eclipse-workspace\\GARF\\FileTrain\\acc_3_0.8_model_11_UpFall_44F.arff"));
        Instances trainData = new Instances(br);
        trainData.setClassIndex(trainData.numAttributes() - 1);
        br.close();

        // setup classifier
        RandomForest rf = new RandomForest();
        Random random = new Random(1); // seed = 1

        int nF_numFolds = 5;
        int neS_NumExecutionSlots = 1; // n
        rf.buildClassifier(trainData);
        int numFolds = nF_numFolds;
//        rf.setBagSizePercent(bsP_BagSizePercent);
 //       rf.setNumIterations(nI_NumIterations);
 //       rf.setNumExecutionSlots(neS_NumExecutionSlots);

        Evaluation evaluation = new Evaluation(trainData);

        evaluation.crossValidateModel(rf, trainData, numFolds, random);
        System.out.println(evaluation.toClassDetailsString());
        System.out.println(evaluation.toMatrixString());
        double fMeasure_Max_for = (evaluation.fMeasure(0) + evaluation.fMeasure(4) + evaluation.fMeasure(5) + evaluation.fMeasure(5)) / 4;
        System.out.println("BSC-" + evaluation.fMeasure(0) + "   FKL-" + evaluation.fMeasure(4) + "    FOL-" + evaluation.fMeasure(5) + "   SDL-" + evaluation.fMeasure(11) + "    all-" + evaluation.weightedFMeasure());
//        System.out.println(fMeasure_Max_for);
        return fMeasure_Max_for;
    }

    private static void convertCSV2AARFF(String input, String output) {
        // TODO Auto-generated method stub
        try {
            CSVLoader loader = new CSVLoader();
            loader.setSource(new File(input));
            Instances data = loader.getDataSet();

            // save ARFF
            ArffSaver saver = new ArffSaver();
            saver.setInstances(data);
            saver.setFile(new File(output));
            // saver.setDestination(new File(output));
            saver.writeBatch();
        } catch (Exception e) {

        }
    }

}
