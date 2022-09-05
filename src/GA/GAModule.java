package GA;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Random;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;

public class GAModule {
	private static int[] num_feature = new int[] {4,5,6,7,8};
	private static int [] min_samples_split = new int[] {1,2,3,4,5,6,7,8,10};
	private static int [] min_samples_leaf = new int[] {1,2,3,4,5,6,7,8,10};
	private static int [] max_depth = new int[] {5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30};
	private static int[] gens = new int[5];
	private static double rate_mutation = 0.1;
	private static double rate_cross = 0.7;
	private static double stop_point = 0.999;
	private static double max_parent_point = 0;
	public static void main(String []args) throws Exception {	
		Random rn =  new Random();	
		//RandomForest(8, 26, 20);
		//Tìm ra 80 cặp bố mẹ tốt nhất
		ArrayList<int[]> Parents_Array =  FindSelectionParent();
		System.out.println("Quan the ban dau chua chon" + Parents_Array.size());
		ArrayList<int[]> Parents_Array_Clone = (ArrayList<int[]>)Parents_Array.clone();
		//System.out.println("So luong quan the ban dau" + Parents_Array.size());
		ArrayList<int[]> Parent = new ArrayList<int[]>();
		ArrayList<int[]> Parent_Clone = new ArrayList<int[]>();		
		ArrayList<int[]> New_Gen = new ArrayList<int[]>();
		ArrayList<int[]> arr_child = new ArrayList<int[]>();
		double point = 0.0;
		int count=0;
		
		//Kiem tra do dai cua gen moi
		 FileWriter fileWriter = new FileWriter("C:\\Users\\dongnv\\eclipse-workspace\\GARF\\src\\GA\\result_max.txt");
		 for (int turn = 0; count <= 20000; turn ++ ) {
			 
			   while (New_Gen.size() < 80) {	 
				// Random chon bo me		
				 for (int i = 0; i < Parents_Array.size(); i++) {
					    float rate_value = rn.nextFloat();	
						// kiem tra rate_selection
						if (rate_value < rate_cross) {
							Parent_Clone.add(Parents_Array.get(i));	
							Parents_Array.remove(i);
						}
						if (Parent_Clone.size() == 2) {
							Parent = Parent_Clone;
							arr_child = CrossOverParent(Parent.get(0),Parent.get(1));
							System.out.println("tap tham so thu nhat " + Parent.get(0)[0] + " " + Parent.get(0)[1] + " " + Parent.get(0)[2]);
							System.out.println("tap tham so thu hai " + Parent.get(1)[0] + " " + Parent.get(1)[1] + " " + Parent.get(1)[2]);
							Parent_Clone.clear();
							New_Gen.add(arr_child.get(0));
							New_Gen.add(arr_child.get(1));
							arr_child.clear();
						}
					}
				}	
			    Parents_Array.clear();
			    for (int l = 0; l < Parents_Array_Clone.size(); l++) {
			    	Parents_Array.add(Parents_Array_Clone.get(l));
			    }
			 
			    PrintWriter printWriter = new PrintWriter(fileWriter);
				for (int f = 0 ; f < New_Gen.size(); f++) {
					System.out.println("Tap con duoc sinh ra " + New_Gen.get(f)[0] + " " +New_Gen.get(f)[1] + " " + New_Gen.get(f)[2]);
					point = RandomForest(New_Gen.get(f)[0],New_Gen.get(f)[1],New_Gen.get(f)[2]);	
					count++;
					if (point > max_parent_point) {
						    printWriter.printf("Tap gia tri con tot hon tim duoc la " + New_Gen.get(f)[0] + " " +New_Gen.get(f)[1] + " " + New_Gen.get(f)[2]);
						    printWriter.printf("Gia tri f1 " + point);
						   
					}
					if (point >= stop_point) {
						System.out.println("Ket thuc chuong trinh");
						System.out.println("Tap gia tri can tim la " + New_Gen.get(f)[0] + " " +New_Gen.get(f)[1] + " " + New_Gen.get(f)[2] );
						System.out.println("F1 thu duoc" + point);
						break;				
					}
				}	
				New_Gen.clear();
				printWriter.close();
		 }
		 
	}
	
	public static double TargetFunction(double arr[]) {
		double rs = -1/5;
		for (int  i = 0; i < arr.length ; i++) {
			rs = rs * Math.pow((1-arr[i]),2) * Math.log(arr[i]);
		}
		return rs;
	};
	
	//Xay dung mo hinh Randomforest
    private static double RandomForest(int max_features, int max_dept,  int nI_NumTree) throws Exception {
        BufferedReader br = null;  
//        convertCSV2AARFF(
//        		"C:\\Users\\dongnv\\Desktop\\acc_3_0.5_model_11_UpFall_48F.csv", 
//        		"C:\\Users\\dongnv\\Desktop\\acc_3_0.5_model_11_UpFall_48F.arff");
        // training file
       // br = new BufferedReader(new FileReader("C:\\Users\\dongnv\\eclipse-workspace\\GARF\\FileTrain\\acc_3_0.8_model_11_UpFall_44F.arff"));
        br = new BufferedReader(new FileReader("C:\\Users\\dongnv\\eclipse-workspace\\GARF\\FileTrain\\acc_3_0.8_model_11_UpFall_44F.arff"));
        Instances trainData = new Instances(br);
        trainData.setClassIndex(trainData.numAttributes() - 1);
        br.close();
        // setup classifier
        RandomForest rf = new RandomForest();
        rf.buildClassifier(trainData);
        Random random = new Random(1); // seed = 1
        int nF_numFolds = 5;
        //int neS_NumExecutionSlots = 1; // n
        rf.setMaxDepth(max_dept);
        rf.setNumFeatures(max_features);  
        //rf.setNumTrees(nI_NumTree);
        //rf.setNumExecutionSlots(neS_NumExecutionSlots);  
        int numFolds = nF_numFolds; 
        Evaluation evaluation = new Evaluation(trainData);
        evaluation.crossValidateModel(rf, trainData, numFolds, random);
        System.out.println(evaluation.toClassDetailsString());
        System.out.println(evaluation.toMatrixString());
        double fMeasure_Max_for = (evaluation.fMeasure(0) + evaluation.fMeasure(1) + evaluation.fMeasure(2) + evaluation.fMeasure(3)+ evaluation.fMeasure(4)) / 5;
        double []targer_arr = {evaluation.fMeasure(0), evaluation.fMeasure(1), evaluation.fMeasure(2), evaluation.fMeasure(3), evaluation.fMeasure(4)};
        double result_target = TargetFunction(targer_arr); 
        System.out.println("A01-" +evaluation.fMeasure(0)  + "   A02-" + evaluation.fMeasure(1) + "    A03-" + evaluation.fMeasure(2) + "   A04-" + evaluation.fMeasure(3) + "   A05-" + evaluation.fMeasure(4) + "    all-" + evaluation.weightedFMeasure());
        for(int i = 0; i < 11; i++) {
        	System.out.println("MCC value of Activity"+ i + evaluation.matthewsCorrelationCoefficient(i) );
        }
        return result_target;
    }

    
    //Ham chon 80 cap gen bo va me    
    public static ArrayList<int[]> FindSelectionParent() throws Exception {
    	Random rn = new Random();
    	int [] arr_selection_max = new int[3];	
    	int [] estimators = n_estimators();
    	ArrayList<int[]> arr_selections_list = new ArrayList<int[]>() ;
    	ArrayList<int[]> arr_selections = new ArrayList<int[]>() ;
    	ArrayList<Double> arrListResult = new ArrayList<Double>();
    	for (int count = 0; count < 100; count ++) {
    		int numFeature = num_feature[rn.nextInt(num_feature.length)];
    		int maxDepth = max_depth[rn.nextInt(max_depth.length)];
    		int nEstimators = estimators[rn.nextInt(estimators.length)];
    		int [] arr_parameter = new int [] {numFeature, maxDepth, nEstimators};
    		System.out.println(count);
    		System.out.println("A group of parameters" + numFeature + " " +maxDepth + " " + nEstimators);
    		double rs = RandomForest(numFeature, maxDepth, nEstimators);
    		arr_selections_list.add(arr_parameter);
    		arrListResult.add(rs);
    	}
    	for (int m = 0; m < arrListResult.size(); m++) {
    		System.out.println("F1 " + arrListResult.get(m));
    	}
    	max_parent_point = FindMaxValue(arrListResult);
    	
    	for (int n = 0; n < 80; n++) {
    		int max_index = FindMax(arrListResult);
        	arr_selection_max = arr_selections_list.get(max_index);
        	arr_selections.add(arr_selection_max);
        	arrListResult.remove(max_index);
    	}
    	return arr_selections;	
    }
    
    public static double FindMaxValue(ArrayList<Double> arr) {
    	double Max = arr.get(0);
    	for (int i = 1; i < arr.size(); i++) {
    		if (arr.get(i) > Max) {
    			Max = arr.get(i);
    		}
    	}
    	return Max;
    }
    
    public static int FindMax(ArrayList<Double> arr) {
    	double Max = arr.get(0);
    	int MaxIndex = 0;
    	for (int i = 0; i < arr.size(); i++) {
    		if (arr.get(i) > Max) {
    			Max = arr.get(i);
    			MaxIndex = i;
    		}
    	}
    	return MaxIndex;
    }
    
    public static String CrossChildLeft(String arr_1, String  arr_2, int i) {
    	String arr_child;
    	arr_child = arr_1.substring(0,i)+ arr_2.substring(i,arr_2.length());
    	return arr_child;	
    }
    
    public static String CrossChildRight(String arr_1, String  arr_2, int i) {
    	String arr_child;
    	arr_child = arr_2.substring(0,i)+ arr_1.substring(i,arr_1.length());
    	return arr_child;	
    }
       
    public static ArrayList<int[]> CrossOverParent(int[] arr1, int[] arr2) {
//    	   for (int k = 0;  k < arr1.length; k++) {
//    		   System.out.print("Tham so cua bo" + arr1[k]);
//    	   }
		   Random rn = new Random();
		   ArrayList<int[]> gen_total = new ArrayList<int[]>();
		   int[] new_gen_1 = new int[arr1.length];
		   int[] new_gen_2 = new int[arr2.length];
		   String  child_1 ;
		   String  child_2 ;
		   int child_1_int;
		   int child_2_int;
	       for (int i = 0; i < arr1.length; i++) {
	    	   String str_att_1 = NewBinanryGens(arr1[i]);
	    	   String str_att_2 = NewBinanryGens(arr2[i]);
	    	   int len_str_1 = str_att_1.length();
	    	   int len_str_2 = str_att_2.length();
	    	   int point;
	    	   if (len_str_1 < len_str_2 ) {
	    		    point = len_str_1;
	    	   }
	    	   else {
	    		    point = len_str_2;
	    	   }
	    	   for (int k = 0; k < point-1; k++ ) { 
	    		   float rn_rate = rn.nextFloat();
	    		   if (rn_rate < rate_cross) {
	    			   child_1 =  CrossChildLeft(str_att_1,str_att_2,k);
	    			   child_2 =  CrossChildRight(str_att_1,str_att_2,k);	    			   
	    			   //mutation 	    			   
	    			   // child 1
	    			   char[] new_child_1 = child_1.toCharArray();
	    			   for(int m = 0; m < child_1.length(); m++) {
	    				   
	    				   float mutation_rate_1 = rn.nextFloat();
	    				  
	    				   if ( mutation_rate_1 < rate_mutation && child_1.charAt(m) == '0') {	    					   
	    					   new_child_1[m] = '1';
	    				   }
	    				   else if ( mutation_rate_1 < rate_mutation && child_1.charAt(m) == '1') {
	    					   new_child_1[m] = '0';
	    				   }
	    			   }	    		
	    			   child_1 = String.valueOf(new_child_1);
	    			   
	    			  // child 2 
	    			   char[] new_child_2 = child_2.toCharArray();
	    			   for(int m = 0; m < child_2.length(); m++) {			   
	    				   float mutation_rate_2 = rn.nextFloat();
	    				   if ( mutation_rate_2 < rate_mutation && child_2.charAt(m) == '0') {	    					   
	    					   new_child_2[m] = '1';
	    				   }
	    				   else if ( mutation_rate_2 < rate_mutation && child_2.charAt(m) == '1') {
	    					   new_child_2[m] = '0';
	    				   }
	    					   
	    			   }	   
	    			   child_2 = String.valueOf(new_child_2);
	    			   
	    			   // String to int
	    			   child_1_int = NewDemicalGens(child_1);
	    			   child_2_int = NewDemicalGens(child_2);	    			   
	    			   //System.out.println("chuoi con he so 10 " + child_1_int);	    			   
	    			   // add att of gen
	    			   new_gen_1[i] = child_1_int;
	    			   new_gen_2[i] = child_2_int;
	    		   }
	    	   }
	       }
	       // add new generation
	       gen_total.add(new_gen_1);
	       gen_total.add(new_gen_2);
	       return gen_total;
	}
	   
    public static String NewBinanryGens(int arr) {
		String str;
		str = Integer.toBinaryString(arr);
		return str;
	}
    
	public static int NewDemicalGens(String str) {
		int arr = 0;
		arr = Integer.parseInt(str,2);
		return arr;
	}
    
    //Lai ghep giua hai ca the
	public static int[] CrossOver(int[] arr1, int[] arr2) {
		   Random rn = new Random();
	        //Select a random crossover point
	        int crossOverPoint = rn.nextInt(arr1.length);
	        int[] arr_cross = new int[arr1.length];
	        //Swap values among parents
	        if (crossOverPoint == 0) {
	        	crossOverPoint = crossOverPoint+1;
	        }
	        for (int i = 0; i < crossOverPoint; i++) {
	        	arr_cross[i] = arr1[i];
	        }
	        for (int j = crossOverPoint; j < arr_cross.length; j++) {
	        	arr_cross[j] = arr2[j];
	        }
	        return arr_cross;
	}
	
	 
	//Ham dot bien
		public void Mutation(String arr[],int i) {
			if (arr[i] == "0") {
				arr[i] = "1";
			}
			else arr[i]="0";
		}

	
	// Kiem ham muc tieu
		protected static boolean fitness(int[] x) {
		        //System.out.println("bsP_BagSizePercent: " + x[0] + " nI_NumIterations: " + x[1]);
		        double fitness = 0;
		        boolean status = false;
		        if (fitness < 0.85)  {
		        	   try {
		   	            fitness = RandomForest(x[0], x[1], x[2]);
		   	        } catch (Exception ex) {
		   	        	System.out.print(ex.toString());
		   	        }	       
		        }
		        else { 
		        	status = true;
		        	System.out.println("Kết quả chỉ số F1 đạt yêu cầu là" + fitness);
		        	System.out.println("Bộ tham số được sử dụng là " + x[0] + " " + x[1]+ " " + x[2]);
		        }
		     
		        System.out.println(fitness);
		        System.out.println("--------------------------------------------------------");
		        System.out.println("");
		        return status;
		}
		
	//Chuyen doi sang nhi phan
	public static String[] BinanryGens(int[] arr) {
		String [] str = new String[3];
		for (int i = 0; i < arr.length; i++) {
			str[i] = Integer.toBinaryString(arr[i]);
		}
		return str;
	}
	
	//Chuyen doi sang thap phan
	public static int[] DemicalGens(String[] str) {
		int [] arr = new int[5];
		for (int i = 0; i < str.length; i++) {
			arr[i] = Integer.parseInt(str[i],2);
		}
		return arr;
	}
	
	
	//Tao estimators
	public static int[] n_estimators() { 
		int[] arr = new int[500] ;
		for (int i = 0; i < 490; i++) {
			arr[i] = i + 10;
		}
		return arr; 
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
	
	
		
		
	 public static double AvaNamdaGocQuay(double []x, double []y, double [] z) {
		 ArrayList<Double> arr_namda = new ArrayList<Double>();
		 	for (int i = 0; i<x.length;i++) {
		 		 double namda = Math.atan(-x[i]/z[i]);
		 		 arr_namda.add(namda);
		 	}
			return Avarage(arr_namda);
	 }
	 
	 public static double VarNamdaGocQuay(double []x, double []y, double [] z) {
		 ArrayList<Double> arr_namda = new ArrayList<Double>();
		 	for (int i = 0; i<x.length;i++) {
		 		 double namda = Math.atan(-x[i]/z[i]);
		 		 arr_namda.add(namda);
		 	}
			return Variance(arr_namda);
	 }
		
		
	 public static double AvaGocQuay(double []x, double []y, double [] z) {
		    ArrayList<Double> arr_phi = new ArrayList<Double>();
		 	for (int i = 0; i<x.length;i++) {
		 		double phiAngle = Math.atan(y[i]/(Math.sqrt(Math.pow(x[i], 2)+ Math.pow(z[i], 2))));
		 		arr_phi.add(phiAngle);
		 	}
			return Avarage(arr_phi);
	 }
	 
	 public static double VarGocQuay(double []x, double []y, double [] z) {
		    ArrayList<Double> arr_phi = new ArrayList<Double>();
		   
		 	for (int i = 0; i<x.length;i++) {
		 		double phiAngle = Math.atan(y[i]/(Math.sqrt(Math.pow(x[i], 2)+ Math.pow(z[i], 2))));
		 		arr_phi.add(phiAngle);
		 	}
			return Variance(arr_phi);
	 }
	 
	 public static double Avarage(ArrayList<Double> arr) {
		 double s = 0;
		 for (int i = 0; i < arr.size(); i++) {
			 s = s + arr.get(i);
		 }
		 return s/(arr.size());
	 }
	 
	 public static double Variance(ArrayList<Double> arr) {
		 double ava =  Avarage(arr);
		 double s = 0;
		 for (int i = 0; i < arr.size(); i++) {
			 s = s + Math.sqrt(arr.get(i)-ava);
		 }
		 return s/(arr.size());
	 }
	 
}



//Lai tạo bố mẹ tạo con
//int[] arr_child = CrossOver(Parents_Array.get(0),Parents_Array.get(1));

//Kiểm tra trạng thái hàm fitness
//boolean fit = fitness(arr_child);
//int turn = 0;
//while(turn < 200)
//	if (fit == true) {
//		System.out.println("Chương trình hoàn thành");
//		break;
//	}
//	else { 
//		System.out.println("Lần chạy thứ" + turn);
//		// Chuyển đổi sang binary string, các tham số giờ ở dạng string
//		String[] str_child_arr = BinanryGens(arr_child);
//		
//		// Duyet tung tham so
//		for (int k = 0; k < str_child_arr.length; k++) {
//			//Tính số phần tử cần đột biến theo rate_mutation  = 0.5 và mỗi lần đột biến sử dụng hàm random
//			for (int point_mutation_count = 0; point_mutation_count < str_child_arr[k].length()* rate_mutation ; point_mutation_count++) {
//					int index_mutation = rn.nextInt(str_child_arr[k].length());
//					StringBuilder plainText = new StringBuilder(str_child_arr[k]);
//					if(str_child_arr[k].charAt(index_mutation) == 0) {	
//						char replace_charactor = '1';
//						plainText.setCharAt(index_mutation, replace_charactor);
//					}
//					else {
//						char replace_charactor = '0';
//						plainText.setCharAt(index_mutation, replace_charactor);
//					}
//			}
//		}
//	//chuyển đổi từ string lại về int
//	int [] new_child_arr = DemicalGens(str_child_arr);
//	fit = fitness(new_child_arr);
//	turn++;
//	
//}