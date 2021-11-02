import java.lang.Math;
import java.util.*;
import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;

public class kNN_test {
    public static void main(String[] args) throws IOException {
        System.out.println("Don't forget to update the optimal k value in this code.");
        System.out.println("DO NOT modify any other line except the k value update.");

        int TRAIN_SIZE=400; //no. training patterns
        int TEST_SIZE=200; //no. validation patterns
        int FEATURE_SIZE=61; //no. of features

        double[][] train = new double [TRAIN_SIZE][FEATURE_SIZE];
        double[][] val = new double [TEST_SIZE][FEATURE_SIZE];
        int[] train_label=new int[TRAIN_SIZE]; //actual target/class label for train data
        int[] val_label=new int[TEST_SIZE]; //actual target/class label for validation data

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

        String val_file="test_data.txt"; //read test data
        try (Scanner tmp = new Scanner(new File(val_file))) {
            for (int i=0; i<TEST_SIZE; i++) {
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

        int NUM_NEIGHBOUR = 21; //UPDATE the k value!!!

        double[][] dist_label = new double[TRAIN_SIZE][2]; //distance array, no of columns+1 to accomodate distance
        double[] y = new double[FEATURE_SIZE];
        double[] x = new double[FEATURE_SIZE];
        int[] predicted_class = new int[TEST_SIZE];
        int[] neighbour = new int[NUM_NEIGHBOUR];

        for (int j=0; j<TEST_SIZE; j++) {//for every test data
            for (int f=0; f<FEATURE_SIZE; f++)
                y[f]=val[j][f];

            for (int i=0; i<TRAIN_SIZE; i++) {
                for (int f=0; f<FEATURE_SIZE; f++)
                    x[f]=train[i][f];

                double sum=0.0;
                for (int f=0; f<FEATURE_SIZE; f++)
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
        for (int j=0; j<TEST_SIZE; j++)
            System.out.print(predicted_class[j] + " ");

        try{
            PrintWriter writer = new PrintWriter("kNN_output.txt", "UTF-8");
            for(int j=0; j<TEST_SIZE; j++)
                writer.print(predicted_class[j] + " ");
            writer.close();
        }
        catch(Exception e)
        {
            System.out.println(e);
        }

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