import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.*;

public class kNN_GA {

    public static int TRAIN_SIZE = 400; //no. training patterns
    public static int VAL_SIZE = 200; //no. validation patterns
    public static int FEATURE_SIZE = 61; //no. of features
    public static double[][] train = new double[TRAIN_SIZE][FEATURE_SIZE]; //data to train
    public static double[][] val = new double[VAL_SIZE][FEATURE_SIZE]; //validation data
    public static int[] train_label = new int[TRAIN_SIZE]; //actual target/class label for train data
    public static int[] val_label = new int[VAL_SIZE]; //actual target/class label for validation data

    public static double accuracy = 0.0;
    public static final double CROSSOVER_PROBABILITY = 0.8;
    public static final double MUTATION_PROBABILITY = 0.1;
    public static final int TOURNAMENT_SELECTION_SIZE = 5;

    public static final int POP_SIZE = 100; //population size, DO NOT MODIFY
    public static final int MAX_GEN = 50; //maximum generation, DO NOT MODIFY

    public static final int NUMBER_OF_ELITE_CHROMOSOME = 20;
    public static Random random = new Random();

    public static void main(String[] args) throws IOException {

        System.out.println("Modify the GA function to obtain the best features.");
        System.out.println("Use your optimal k value for kNN that you found for Problem 1");
        System.out.println("DO NOT modify the MAX_POP or MAX_GEN constants or the output file generation");

        Load_Data(TRAIN_SIZE, VAL_SIZE, FEATURE_SIZE); //load data
        GA(); //call GA function

    } //end main loop


    public static void Sort(double[][] sort_array, final int column_sort) {
       /*System.setProperty("java.util.Arrays.useLegacyMergeSort", "true"); */

        Arrays.sort(sort_array, new Comparator<double[]>() {
            @Override
            public int compare(double[] a, double[] b) {
                if ((a[column_sort - 1] > b[column_sort - 1] ) || (a[column_sort - 1] == b[column_sort - 1]))
                    return 1;
                else return -1;
            }
        });
    }

    public static int Mode(int neigh[]) {
        int modeVal = 0;
        int maxCnt = 0;

        for (int i = 0; i < neigh.length; ++i) {
            int count = 0;
            for (int j = 0; j < neigh.length; ++j) {
                if (neigh[j] == neigh[i])
                    count = count + 1;
            }
            if (count > maxCnt) {
                maxCnt = count;
                modeVal = neigh[i];
            }
        }
        return modeVal;
    }


    public static void Load_Data(int TRAIN_SIZE, int VAL_SIZE, int FEATURE_SIZE) throws IOException {

        String train_file = "train_data.txt"; //read training data
        try (Scanner tmp = new Scanner(new File(train_file))) {
            for (int i = 0; i < TRAIN_SIZE; i++)
                for (int j = 0; j < FEATURE_SIZE; j++)
                    if (tmp.hasNextDouble())
                        train[i][j] = tmp.nextDouble();
            tmp.close();
        }

        String val_file = "val_data.txt"; //read validation data
        try (Scanner tmp = new Scanner(new File(val_file))) {
            for (int i = 0; i < VAL_SIZE; i++)
                for (int j = 0; j < FEATURE_SIZE; j++)
                    if (tmp.hasNextDouble())
                        val[i][j] = tmp.nextDouble();
            tmp.close();
        }

        String train_label_file = "train_data_label.txt"; //read train label
        try (Scanner tmp = new Scanner(new File(train_label_file))) {
            for (int i = 0; i < TRAIN_SIZE; i++)
                if (tmp.hasNextInt())
                    train_label[i] = tmp.nextInt();
            tmp.close();
        }

        String val_label_file = "val_data_label.txt"; //read validation label (to obtain classification accuracy)
        try (Scanner tmp = new Scanner(new File(val_label_file))) {
            for (int i = 0; i < VAL_SIZE; i++)
                if (tmp.hasNextInt())
                    val_label[i] = tmp.nextInt();
            tmp.close();
        }

    }

    private static boolean[][] crossover(boolean[] p1, boolean[] p2) {
        boolean[][] crossedOverChildren = new boolean[2][FEATURE_SIZE];
        double randomProbability = random.nextDouble();

        if (randomProbability < CROSSOVER_PROBABILITY) {
            boolean multipointCrossover = random.nextBoolean();
            int crossOverPoint1 = random.nextInt(FEATURE_SIZE);

            int end=Integer.MIN_VALUE;
            if(multipointCrossover){
                end = random.nextInt(FEATURE_SIZE);
            }else{
                end=p1.length;
            }

            crossedOverChildren[0] = p1;
            crossedOverChildren[1] = p2;

            for(int i=crossOverPoint1;i<end;i++){
                crossedOverChildren[0][i] = p2[i];
                crossedOverChildren[1][i]=  p1[i];
            }

        } else {
            crossedOverChildren[0] = p1;
            crossedOverChildren[1] = p2;
        }
        return crossedOverChildren;
    }

    private static boolean[] mutation(boolean[] child) {
        boolean[] mutatedChild = null;

        Double randomProbability = random.nextDouble();
        mutatedChild = child;

        if (randomProbability < MUTATION_PROBABILITY) {
            int mutationPoint = random.nextInt(FEATURE_SIZE);
            mutatedChild[mutationPoint] = !(mutatedChild[mutationPoint]);
        }
        return mutatedChild;
    }

    private static boolean[][] elitism(boolean[][] sol, double fitness[]){

        boolean[][] elitePopulation = new boolean[POP_SIZE][FEATURE_SIZE];
        double max = 0;
        int index;
        for (int j = 0; j < NUMBER_OF_ELITE_CHROMOSOME; j++) {

            max = fitness[0];
            index = 0;
            for (int i = 0; i < fitness.length; i++) {
                if (max < fitness[i]) {
                    max = fitness[i];
                    index = i;
                }
            }
            elitePopulation[j] = sol[index];

            fitness[index] = 0;

        }
        return elitePopulation;
    }

    private static boolean[][] evolve(boolean[][] sol, double fitness[]) {


        boolean[][] finalNewGeneration = new boolean[POP_SIZE][FEATURE_SIZE];
        boolean[][] temp_elitePopulation = null;
        boolean[][] temp_tournamentPopulation = new boolean[POP_SIZE][FEATURE_SIZE];


        //getting elite population from current generation
        temp_elitePopulation=elitism(sol,fitness);
        if (temp_elitePopulation != null) {

            for(int i=0;i<NUMBER_OF_ELITE_CHROMOSOME;i++){
                finalNewGeneration[i]=temp_elitePopulation[i];
            }
        }

        //getting remaining population from tournament selection
        for (int i = 0; i < POP_SIZE-NUMBER_OF_ELITE_CHROMOSOME; i++) {
            temp_tournamentPopulation[i] = tournamentSelection(sol, fitness);
        }


        //check if parent1 and parent2 belong to indices in fit population , regenerate
        int incrementer=NUMBER_OF_ELITE_CHROMOSOME;
        for(int i=0; i<POP_SIZE/2 && incrementer<POP_SIZE; i++){

            int parent1 = random.nextInt(POP_SIZE - 1);
            int parent2=parent1;
            //to ensure different parents are selected
            while(parent2==parent1){
                parent2 = random.nextInt(POP_SIZE - 1);
            }

            boolean[][] crossedOverChildren = crossover(temp_tournamentPopulation[parent1], temp_tournamentPopulation[parent2]);

            for (int j = 0; j < crossedOverChildren.length && incrementer<POP_SIZE; j++) {
                finalNewGeneration[incrementer]=mutation(crossedOverChildren[j]);
                incrementer++;
            }

        }

        return finalNewGeneration;
    }

    private static boolean[] tournamentSelection(boolean[][] sol, double[] fitness) {
        int maxAt = 0;
        boolean[][] tournamentSelectionPopulation = new boolean[TOURNAMENT_SELECTION_SIZE][FEATURE_SIZE];
        double[] tournamentFitness = new double[TOURNAMENT_SELECTION_SIZE];

        for (int i = 0; i < TOURNAMENT_SELECTION_SIZE; i++) {
            int selectionIndex = random.nextInt(POP_SIZE - 1);
            tournamentSelectionPopulation[i] = sol[selectionIndex];
            tournamentFitness[i] = fitness[selectionIndex];
        }
        for (int j = 0; j < TOURNAMENT_SELECTION_SIZE; j++) {
            maxAt = tournamentFitness[j] > tournamentFitness[maxAt] ? j : maxAt;
        }
        return tournamentSelectionPopulation[maxAt];
    }

    public static boolean[] GA() {

        boolean[] temp_sol = new boolean[FEATURE_SIZE];
        double[] fitness = new double[POP_SIZE];
        boolean[][] sol = new boolean[POP_SIZE][FEATURE_SIZE];
        boolean[][] new_sol = new boolean[POP_SIZE][FEATURE_SIZE];
        boolean[] final_sol = new boolean[FEATURE_SIZE]; //final best pop

        //create initial population
        for (int j = 0; j < POP_SIZE; j++) {
            int count = 0;
            for (int k = 0; k < FEATURE_SIZE; k++) {
                sol[j][k] = (Math.random() > 0.5);
                temp_sol[k] = sol[j][k];
                if (temp_sol[k] == true)
                    count++;
            }
            //modify fitness to include both increasing accuracy and minimising features
            if (count > 40)
                fitness[j] = 0.0;
            else
                fitness[j] = KNN(train, val, train_label, val_label, temp_sol) - (count / FEATURE_SIZE);

            //System.out.print(count + " ");
        }
        //System.out.println();

        new_sol = sol; //copy initial array


        for (int gen = 0; gen < MAX_GEN; gen++) { //do for many generations

            sol = new_sol; //parent copied as children for GA algorithm

            sol = evolve(sol, fitness);

            //compute fitness
            for (int j = 0; j < POP_SIZE; j++) {
                int count = 0;
                for (int k = 0; k < FEATURE_SIZE; k++) {
                    temp_sol[k] = sol[j][k];
                    if (temp_sol[k] == true)
                        count++;
                }
                if (count > 40)
                    fitness[j] = 0.0;
                else {
                    fitness[j] = KNN(train, val, train_label, val_label, temp_sol) - (count / FEATURE_SIZE);
                }

                System.out.print(fitness[j] + " ");
            }


            int maxAt = 0;
            for (int j = 0; j < POP_SIZE; j++)
                maxAt = fitness[j] > fitness[maxAt] ? j : maxAt;
            for (int k = 0; k < FEATURE_SIZE; k++)
                final_sol[k] = sol[maxAt][k];
            System.out.println("Best fitness = " + fitness[maxAt]);

            //update the population
            //copy fitter children as parent for next generation
            new_sol = sol;

        } //end of gen loop

        int channel = 0;
        for (int k = 0; k < FEATURE_SIZE; k++)
            if (final_sol[k]) {
                System.out.print(1 + " ");
                channel++;
            } else
                System.out.print(0 + " ");
        System.out.println("Channel count =  " + channel);

        System.out.println("Accuracy =  " + KNN(train, val, train_label, val_label, final_sol));


        //write best solution (features) to file, DO NOT MODIFY
        try {
            PrintWriter writer = new PrintWriter("kNN_GA_output.txt", "UTF-8");
            for (int j = 0; j < FEATURE_SIZE; j++)
                if (final_sol[j])
                    writer.print("1 ");
                else
                    writer.print("0 ");
            writer.close();
        } catch (Exception e) {
            System.out.println(e);
        }

        return final_sol;

    }


    public static double KNN(double[][] train, double[][] val, int[] train_label, int[] val_label, boolean[] sol) {
        double[][] dist_label = new double[TRAIN_SIZE][2]; //distance array, no of columns+1 to accomodate distance
        double[] y = new double[FEATURE_SIZE];
        double[] x = new double[FEATURE_SIZE];

        int num_neighbour = 21; //optimal k value

        int[] neighbour = new int[num_neighbour];
        int[] predicted_class = new int[VAL_SIZE];

        for (int j = 0; j < VAL_SIZE; j++) {//for every validation data
            for (int f = 0; f < FEATURE_SIZE; f++)
                if (sol[f])
                    y[f] = val[j][f];
                else
                    y[f] = 0.0;

            for (int i = 0; i < TRAIN_SIZE; i++) {
                for (int f = 0; f < FEATURE_SIZE; f++)
                    if (sol[f])
                        x[f] = train[i][f];
                    else
                        x[f] = 0.0;

                double sum = 0.0; //Euclidean distance
                for (int f = 0; f < FEATURE_SIZE; f++)
                    sum = sum + ((x[f] - y[f]) * (x[f] - y[f]));

                dist_label[i][0] = Math.sqrt(sum);
                dist_label[i][1] = train_label[i];
            }

            Sort(dist_label, 1); //Sorting distance

            for (int n = 0; n < num_neighbour; n++) //training label from required neighbours
                neighbour[n] = (int) dist_label[n][1];

            predicted_class[j] = Mode(neighbour);

        } //end val data loop

        int success = 0;
        for (int j = 0; j < VAL_SIZE; j++)
            if (predicted_class[j] == val_label[j])
                success++;
        accuracy = (success * 1.0) / VAL_SIZE;
        //System.out.print(accuracy + " ");

        return accuracy;
    }

} //end class loop