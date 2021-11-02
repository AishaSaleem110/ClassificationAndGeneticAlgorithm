import java.lang.Math;
import java.util.*;
import java.io.File;
import java.io.IOException;


public class kNN {
    public static void main(String[] args) throws IOException {
        System.out.println("These are instructions of how to solve the problems. Please make sure you follow the instructions carefully.");
        System.out.println("Use this function to obtain the optimal k value and then include this k value in the kNN_test.java file. ");

        int TRAIN_SIZE=400; //no. training patterns
        int VAL_SIZE=200; //no. validation patterns
        int FEATURE_SIZE=61; //no. of features

        double[][] train = new double [TRAIN_SIZE][FEATURE_SIZE];
        double[][] val = new double [VAL_SIZE][FEATURE_SIZE];
        int[] train_label=new int[TRAIN_SIZE]; //actual target/class label for train data
        int[] val_label=new int[VAL_SIZE]; //actual target/class label for validation data

        String train_file="train_data.txt"; //read training data
        try (Scanner tmp = new Scanner(new File(train_file))) {
            for (int i=0; i<TRAIN_SIZE; i++) {
                for (int j=0; j<FEATURE_SIZE; j++) {
                    if(tmp.hasNextDouble()) {
                        train[i][j]=tmp.nextDouble();
                        //System.out.print(train[i][j] + " ");
                    }
                }
            }
            tmp.close();
        }

        String val_file="val_data.txt"; //read validation data
        try (Scanner tmp = new Scanner(new File(val_file))) {
            for (int i=0; i<VAL_SIZE; i++) {
                for (int j=0; j<FEATURE_SIZE; j++) {
                    if(tmp.hasNextDouble()) {
                        val[i][j]=tmp.nextDouble();
                        //System.out.print(val[i][j] + " ");
                    }
                }
            }
            tmp.close();
        }

        String train_label_file="train_data_label.txt"; //read train label
        try (Scanner tmp = new Scanner(new File(train_label_file))) {
            for (int i=0; i<TRAIN_SIZE; i++) {
                if(tmp.hasNextInt()) {
                    train_label[i]=tmp.nextInt();
                    //System.out.print(train_label[i] + " ");
                }
            }
            tmp.close();
        }

        //read validation label (to obtain classification accuracy)
        String val_label_file="val_data_label.txt";
        try (Scanner tmp = new Scanner(new File(val_label_file))) {
            for (int i=0; i<VAL_SIZE; i++) {
                if(tmp.hasNextInt()) {
                    val_label[i]=tmp.nextInt();
                    //System.out.print(val_label[i] + " ");
                }
            }
            tmp.close();
        }

        int NUM_NEIGHBOUR = 21; //k value
        double[][] dist_label = new double[TRAIN_SIZE][2]; //distance array, no of columns+1 to accomodate distance//distance for each

        double[] y = new double[FEATURE_SIZE]; //validation data
        double[] x = new double[FEATURE_SIZE]; //training data
        int[] predicted_class = new int[VAL_SIZE];
        int[] neighbour = new int[NUM_NEIGHBOUR];

        for (int j=0; j<VAL_SIZE; j++) {//for every validation data
            for (int f=0; f<FEATURE_SIZE; f++)
                y[f]=val[j][f];

            for (int i=0; i<TRAIN_SIZE; i++) {
                for (int f=0; f<FEATURE_SIZE; f++)
                    x[f]=train[i][f];

                double sum=0.0;
                for (int f=0; f<FEATURE_SIZE; f++)
                    //Euclidean distance
                    sum=sum + ((x[f]-y[f])*(x[f]-y[f]));

                dist_label[i][0] = Math.sqrt(sum);
                dist_label[i][1] = train_label[i];
                //System.out.println(dist_label[i][0] + " " + dist_label[i][1]);
            }

            Sort(dist_label,1);

            for (int n=0; n<NUM_NEIGHBOUR; n++) //training label from required neighbours
                neighbour[n]=(int) dist_label[n][1];

            predicted_class[j]=Mode(neighbour);
        } //end test data loop


        int success=0;
        System.out.println("Predicted class: ");

        //These codes can only be run for validation data, not test data
        for (int j=0; j<VAL_SIZE; j++){
            System.out.print(predicted_class[j] + " ");
            if (predicted_class[j]==val_label[j])
                success=success+1;
        }
        double accuracy=(success*100.0)/VAL_SIZE;
        System.out.println("Accuracy = " + accuracy);

    } //end main loop


    public static void Sort (double[][] sort_array, final int column_sort) {
        Arrays.sort(sort_array, new Comparator<double[]>() {
            @Override
            public int compare(double[] a, double[] b) {
                if(a[column_sort-1] > b[column_sort-1]) return 1;
                else return -1;
            }
        });
    }

    public static int Mode(int neigh[]) {
        int modeVal=0;
        int maxCnt=0;

        for (int i = 0; i < neigh.length; ++i) {
            int count = 0;
            for (int j = 0; j < neigh.length; ++j) {
                if (neigh[j] == neigh[i])
                    count=count+1;
            }
            if (count > maxCnt) {
                maxCnt = count;
                modeVal = neigh[i];
            }
        }
        return modeVal;
    }


} //end class loop