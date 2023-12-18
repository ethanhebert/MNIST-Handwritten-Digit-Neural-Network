/* 
Ethan Hebert
103-81-833
10/25/23
Assignment 2 Part 2 - MNIST Handwritten Digit Recognizer Neural Network
This program solves the MNIST Handwritten Digit Recognizer problem with a fully connected feed forward neural network
constructed in Java utilizing back propogation and stochastic gradient descent. The program functions within a menu
interface in a terminal which allows a user to train a new network, open a previously saved network, save a network,
and interact with this network statistically and visually.
*/

import java.io.*;
import java.util.*;

class mnistnn {
    // GLOBAL CONSTANTS
    // PREDEFINED VARIABLES
    public static final double ETA = 3.0;
    public static final int LAYERS = 3;
    public static final int[] LAYERSIZES = {784,15,10};
    public static final int MINIBATCHSIZE = 10;
    public static final int EPOCHS = 30;
    public static final int TRAININGDATASIZE = 60000;
    public static final int TESTINGDATASIZE = 10000;
    public static final int IMG_WIDTH = 28;
    public static final File TRAININGDATAFILE = new File("mnist_train.csv");
    public static final File TESTINGDATAFILE = new File("mnist_test.csv");
    public static final Scanner USERINPUTSCANNER = new Scanner(System.in);

    // ANSI COLOR CODES
    public static final String RESET = "\033[0m";
    public static final String RED = "\033[1;31m";
    public static final String GREEN = "\033[1;32m";

    // GLOBAL VARIABLES
    // WEIGHTS - EACH NODE IN A LAYER HAS K WEIGHTS WHERE K IS THE NUMBER OF NODES IN THE PREVIOUS LAYER
    public static double[][] layer1weights = new double[LAYERSIZES[1]][LAYERSIZES[0]];
    public static double[][] layer2weights = new double[LAYERSIZES[2]][LAYERSIZES[1]];

    // BIASES, ACTIVATIONS, AND BIAS GRADIENTS - EACH NODE IN A LAYER HAS ONE OF EACH OF THESE
    public static double[] layer1biases = new double[LAYERSIZES[1]];
    public static double[] layer2biases = new double[LAYERSIZES[2]];
    public static double[] layer1activations = new double[LAYERSIZES[1]];
    public static double[] layer2activations = new double[LAYERSIZES[2]];
    public static double[] layer1biasGradients = new double[LAYERSIZES[1]];
    public static double[] layer2biasGradients = new double[LAYERSIZES[2]];
    public static double[][] layer1weightGradients = new double[LAYERSIZES[1]][LAYERSIZES[0]];
    public static double[][] layer2weightGradients = new double[LAYERSIZES[2]][LAYERSIZES[1]];

    // TRAINING/TESTING DATA - INDICES MAP BETWEEN INPUTS AND OUTPUTS
    public static double[][] trainingInputs = new double[TRAININGDATASIZE][LAYERSIZES[0]];
    public static double[][] trainingOutputs = new double[TRAININGDATASIZE][LAYERSIZES[2]];
    public static double[][] testingInputs = new double[TESTINGDATASIZE][LAYERSIZES[0]];
    public static double[][] testingOutputs = new double[TESTINGDATASIZE][LAYERSIZES[2]];

    // RANDOM ORDERING OF TRAINING INPUTS STORED AS INDICES FOR MINI-BATCHES TO PULL FROM
    public static int[] trainingInputsRandomIndices = new int[TRAININGDATASIZE];

    // NETWORK'S OUTPUTS - GUESS HANDWRITTEN DIGIT
    public static double[][] trainingNetworkOutputs = new double[TRAININGDATASIZE][LAYERSIZES[2]];
    public static double[][] testingNetworkOutputs = new double[TESTINGDATASIZE][LAYERSIZES[2]];

    // TOTAL NUMBER OF EACH HANDWRITTEN DIGIT WITHIN INPUT DATA
    public static int[] totalOfEachDigit = new int[LAYERSIZES[2]];

    // CURRENT LOADED NETWORK
    public static String currentLoadedNetwork = "NONE";

    // The program begins here when called by a terminal.
    public static void main(String[] args) throws IOException {
        // start with the main menu
        displayMenu();
    }

    // The main menu of the program
    public static void displayMenu() throws FileNotFoundException {
        // welcome to the program/instructions
        clearTerminal();
        displayInstructions();

        // get user selection and error check
        int selection = getUserSelectionMenu();

        // perform function based on user selection
        switch (selection) {
            // Train the network
            case 1:
                // clear the menu and notify user of training
                clearTerminal();
                System.out.println("Training network...\n");

                // read training data inputs and outputs from file and put into arrays
                readMNISTData(TRAININGDATAFILE);

                // shuffle the training input data into a random order, stored as an array of indices
                randomizeTrainingInputsIndices();

                // create random initial weights and biases
                createRandomInitialWeights(layer1weights);
                createRandomInitialWeights(layer2weights);
                createRandomInitialBiases(layer1biases);
                createRandomInitialBiases(layer2biases);

                // run neural network
                stochasticGradientDescent();

                // display return instructions and wait for input
                currentLoadedNetwork = "NEWLY TRAINED NETWORK";
                System.out.println("[ENTER] Main Menu\n");
                USERINPUTSCANNER.nextLine();
                displayMenu();

                break;
            // Load a pre-trained network
            case 2:
                // display instructions
                clearTerminal();
                System.out.println("Select a pre-trained network that you wish to load.");
                System.out.println("Enter the number of your selection.\n");

                // get every file in the current directory
                File directory = new File("./");
                String directoryContents[] = directory.list();

                // create array of just the .csv files
                boolean[] isCSV = new boolean[directoryContents.length];
                int totalCSVs = 0;
                for (int i=0; i<directoryContents.length; i++) {
                    if (directoryContents[i].endsWith(".csv")) {
                        isCSV[i] = true;
                        totalCSVs++;
                    }
                    else
                        isCSV[i] = false;
                }
                String[] listOfCSVs = new String[totalCSVs];
                int listOfCSVsIndex = 0;
                for (int i=0; i<isCSV.length; i++) {
                    if (isCSV[i]) {
                        listOfCSVs[listOfCSVsIndex] = directoryContents[i];
                        listOfCSVsIndex++;
                    }
                }
                
                // display just the .csv files with selection numbers
                for (int i=0; i<listOfCSVs.length; i++)
                    System.out.println("[" + (i+1) + "] " + listOfCSVs[i]);
                
                // get user selection
                System.out.println();
                int userSelectionCSV = getUserSelectionCSV(totalCSVs, 0);

                // load the user selected network from the file
                String inputFilename = listOfCSVs[userSelectionCSV-1];
                Scanner loadNetworkScanner = new Scanner(new File(inputFilename));

                // layer 1 weights
                for (int i=0; i<layer1weights.length; i++) {
                    String currLine = loadNetworkScanner.nextLine();
                    String[] currLineSplit = currLine.split(",");
                    for (int j=0; j<layer1weights[0].length; j++)
                        layer1weights[i][j] = Double.parseDouble(currLineSplit[j]);
                }

                // layer 2 weights
                for (int i=0; i<layer2weights.length; i++) {
                    String currLine = loadNetworkScanner.nextLine();
                    String[] currLineSplit = currLine.split(",");
                    for (int j=0; j<layer2weights[0].length; j++)
                        layer2weights[i][j] = Double.parseDouble(currLineSplit[j]);
                }

                // layer 1 biases
                String currLine = loadNetworkScanner.nextLine();
                String[] currLineSplit = currLine.split(",");
                for (int i=0; i<layer1biases.length; i++)
                    layer1biases[i] = Double.parseDouble(currLineSplit[i]);

                // layer 2 biases
                String currLine2 = loadNetworkScanner.nextLine();
                String[] currLine2Split = currLine2.split(",");
                for (int i=0; i<layer2biases.length; i++)
                    layer2biases[i] = Double.parseDouble(currLine2Split[i]);

                loadNetworkScanner.close();

                // give option to return to menu
                currentLoadedNetwork = inputFilename;
                System.out.println(GREEN + inputFilename + " successfully loaded.\n" + RESET);
                System.out.println("[ENTER] Main Menu\n");
                USERINPUTSCANNER.nextLine();
                displayMenu();

                break;
            // Display network accuracy on TRAINING data
            case 3:
                // clear the menu and notify user of training
                clearTerminal();
                System.out.println("Running network on TRAINING data...\n");

                // import the training data into arrays
                readMNISTData(TRAININGDATAFILE);

                // run the current network on the training data without changing any weights
                runNetworkWithCurrentWeights(TRAININGDATAFILE);

                // generate statistics on activations
                generateOutputStatistics(0, 3);

                // display return instructions and wait for input
                System.out.println("[ENTER] Main Menu\n");
                USERINPUTSCANNER.nextLine();
                displayMenu();

                break;
            // Display network accuracy on TESTING data
            case 4:
                // clear the menu and notify user of training
                clearTerminal();
                System.out.println("Running network on TESTING data...\n");

                // import the testing data into arrays
                readMNISTData(TESTINGDATAFILE);

                // run the current network on the testing data without changing any weights
                runNetworkWithCurrentWeights(TESTINGDATAFILE);

                // generate statistics on activations
                generateOutputStatistics(0, 4);

                // display return instructions and wait for input
                System.out.println("[ENTER] Main Menu\n");
                USERINPUTSCANNER.nextLine();
                displayMenu();

                break;
            // Run network on TESTING data showing images and labels
            case 5:
                // clear the menu and notify user of training
                clearTerminal();
                System.out.println("Loading...\n");

                // import the testing data into arrays
                readMNISTData(TESTINGDATAFILE);

                // run the current network on the testing data without changing any weights and print all images
                runNetworkWithImages(false);
                displayMenu();

                break;
            // Display the misclassified TESTING images
            case 6:
                // clear the menu and notify user of training
                clearTerminal();
                System.out.println("Loading...\n");

                // import the testing data into arrays
                readMNISTData(TESTINGDATAFILE);

                // run the current network on the testing data without changing any weights and print only misclassified images
                runNetworkWithImages(true);
                displayMenu();

                break;
            // Save the network state to file
            case 7:
                // display instructions
                clearTerminal();
                System.out.println("The current weight set will be saved to a .csv file.");
                System.out.println("Enter your output file as '<filename>.csv' where you replace <filename> with your file name.\n");

                // get output file name
                String outputFilename = getOutputFilename(1);

                // save to output file
                PrintStream console = System.out;
                PrintStream out = new PrintStream(new File(outputFilename));
                System.setOut(out);

                // layer 1 weights
                for (int i=0; i<layer1weights.length; i++) {
                    for (int j=0; j<layer1weights[0].length; j++) {
                        System.out.print(layer1weights[i][j] + ",");
                    }
                    System.out.println();
                }
                
                // layer 2 weights
                for (int i=0; i<layer2weights.length; i++) {
                    for (int j=0; j<layer2weights[0].length; j++) {
                        System.out.print(layer2weights[i][j] + ",");
                    }
                    System.out.println();
                }

                // layer 1 biases
                for (int i=0; i<layer1biases.length; i++)
                    System.out.print(layer1biases[i] + ",");
                System.out.println();

                // layer 2 biases
                for (int i=0; i<layer2biases.length; i++)
                    System.out.print(layer2biases[i] + ",");
                System.out.println();

                // switch back to console output and display finished
                System.setOut(console);
                System.out.println(GREEN + outputFilename + " saved.\n" + RESET);
                currentLoadedNetwork = outputFilename;

                // give option to return to menu
                System.out.println("[ENTER] Main Menu\n");
                USERINPUTSCANNER.nextLine();
                displayMenu();

                break;
            // Exit
            case 0:
                clearTerminal();
                USERINPUTSCANNER.close();
                break;
        }
    }

    // This function handles error checking for getting user selection input on the main menu.
    public static int getUserSelectionMenu() {
        // get input
        int selection;
        try {
            selection = Integer.parseInt(USERINPUTSCANNER.nextLine());
        }
        catch (Exception e) {
            System.out.println(RED + "Invalid selection.\n" + RESET);
            return getUserSelectionMenu();
        }

        boolean validSelection = false;

        // error check
        if (currentLoadedNetwork == "NONE") {
            if (selection == 0 || selection == 1 || selection == 2)
                validSelection = true;
        }
        else {
            for (int i=0; i<=7; i++) {
                if (selection == i)
                    validSelection = true;
            }
        }

        // return value or call for input again
        if (!validSelection) {
            System.out.println(RED + "Invalid selection.\n" + RESET);
            return getUserSelectionMenu();
        }
        return selection;
    }

    // This function serves menu cases 5 and 6. It creates a sort of interface that displays 
    // digit images and controls to navigate through the images.
    public static void runNetworkWithImages(boolean misclassified) {
        for (int i=0; i<TESTINGDATASIZE; i++) {
            // get activations
            backPropagation(testingInputs[i], testingOutputs[i]);

            // store these activations
            testingNetworkOutputs[i] = layer2activations;

            // find what digit the network guessed
            double max = -1.0;
            int networkGuessedDigit = -1;
            for (int j=0; j<LAYERSIZES[2]; j++) {
                if (testingNetworkOutputs[i][j] > max) {
                    max = testingNetworkOutputs[i][j];
                    networkGuessedDigit = j;
                }
            }

            // find the correct output digit
            int MNISTCorrectDigit = -2;
            for (int j=0; j<LAYERSIZES[2]; j++) {
                if (testingOutputs[i][j] == 1.0) {
                    MNISTCorrectDigit = j;
                    break;
                }
            }

            // if showing all images, compare the two and print header text and image
            if (!misclassified) {
                clearTerminal();
                if (networkGuessedDigit == MNISTCorrectDigit)
                    System.out.println("Testing Case #" + (i+1) + ":  Correct Classification = " + MNISTCorrectDigit + " | Network Output = " + networkGuessedDigit + " | " + GREEN + "CORRECT\n");
                else
                    System.out.println("Testing Case #" + (i+1) + ":  Correct Classification = " + MNISTCorrectDigit + " | Network Output = " + networkGuessedDigit + " | " + RED + "INCORRECT\n");
            }
            // if showing just misclassified
            else {
                if (networkGuessedDigit != MNISTCorrectDigit) {
                    clearTerminal();
                    System.out.println("Testing Case #" + (i+1) + ":  Correct Classification = " + MNISTCorrectDigit + " | Network Output = " + networkGuessedDigit + " | " + RED + "INCORRECT\n");
                }
                else
                    continue;
            }
            
            // print the digit itself
            generateDigitImage(testingInputs[i]);
            System.out.print(RESET);

            // allow user to go to next image or main menu
            System.out.println("\n[ENTER] Next Image");
            System.out.println("[0] Main Menu\n");
            String userInput = USERINPUTSCANNER.nextLine();
            System.out.println(userInput);
            if (userInput.equals("0"))
                break;
        }
    }

    // This function makes ASCII art to display a hand-drawn image from the input data.
    public static void generateDigitImage(double[] pixels) {
        for (int i=0; i<IMG_WIDTH; i++) {
            String line = "";
            for (int j=0; j<IMG_WIDTH; j++) {
                if (pixels[i*IMG_WIDTH + j] >= 0.875)
                    line += "& ";
                else if (pixels[i*IMG_WIDTH + j] >= 0.75)
                    line += "X ";
                else if (pixels[i*IMG_WIDTH + j] >= 0.625)
                    line += "H ";
                else if (pixels[i*IMG_WIDTH + j] >= 0.5)
                    line += "k ";
                else if (pixels[i*IMG_WIDTH + j] >= 0.375)
                    line += "i ";
                else if (pixels[i*IMG_WIDTH + j] >= 0.25)
                    line += "o ";
                else if (pixels[i*IMG_WIDTH + j] >= 0.125)
                    line += ", ";
                else if (pixels[i*IMG_WIDTH + j] > 0.0)
                    line += ". ";
                else {
                    line += "  ";
                }
            }
            System.out.println(line);
        }
    }

    // This function serves menu cases 3 and 4. It runs through either the training or testing data a single epoch,
    // storing the activations as it goes.
    public static void runNetworkWithCurrentWeights(File dataFile) {
        if (dataFile == TRAININGDATAFILE) {
            for (int i=0; i<TRAININGDATASIZE; i++) {
                // get activations for each input
                backPropagation(trainingInputs[i], trainingOutputs[i]);
                // store these activations
                trainingNetworkOutputs[i] = layer2activations;
            }
        }
        else if (dataFile == TESTINGDATAFILE) {
            for (int i=0; i<TESTINGDATASIZE; i++) {
                // get activations for each input
                backPropagation(testingInputs[i], testingOutputs[i]);
                // store these activations
                testingNetworkOutputs[i] = layer2activations;
            }
        }
    }

    // This function serves case 2 on the main menu, loading a network from a .csv file.
    // It handles the invalid file selection error.
    public static int getUserSelectionCSV(int totalCSVs, int attempt) {
        if (attempt > 0) {
            System.out.println(RED + "Invalid selection.\n" + RESET);
        }
        int userSelectionCSV;
        try {
            userSelectionCSV = Integer.parseInt(USERINPUTSCANNER.nextLine());
        }
        catch (Exception e) {
            return getUserSelectionCSV(totalCSVs, ++attempt);
        }
        if (userSelectionCSV <= 0 || userSelectionCSV > totalCSVs)
            return getUserSelectionCSV(totalCSVs, ++attempt);
        return userSelectionCSV;
    }

    // This function serves case 7 on the main menu, saving weights to an output file.
    // This checks for an inputted filename that ends with ".csv".
    public static String getOutputFilename(int attempt) {
        if (attempt > 1) {
            System.out.println(RED + "Invalid filename.\n" + RESET);
        }
        String outputFilename = USERINPUTSCANNER.nextLine();
        if (!outputFilename.endsWith(".csv"))
            return getOutputFilename(++attempt);
        return outputFilename;
    }

    // This function clears the terminal screen.
    public static void clearTerminal() {  
        System.out.print("\033[H\033[2J");  
        System.out.flush();
    }  

    // This function shuffles the indices for training inputs and stores this shuffled index
    // order in a separate array to later randomize mini-batches.
    public static void randomizeTrainingInputsIndices() {
        // fill the array with numbers 0-(TRAININGDATASIZE-1) in order
        for (int i=0; i<TRAININGDATASIZE; i++)
            trainingInputsRandomIndices[i] = i;
        // now shuffle the array
        Random rand = new Random();
        for (int i=0; i<TRAININGDATASIZE; i++) {
            int randomIndex = rand.nextInt(TRAININGDATASIZE);
            int valueAtIndex = trainingInputsRandomIndices[randomIndex];
            trainingInputsRandomIndices[randomIndex] = trainingInputsRandomIndices[i];
            trainingInputsRandomIndices[i] = valueAtIndex;
        }
    }

    // This function prints the main menu instructions/navigation.
    public static void displayInstructions() {
        System.out.println("Welcome to the MNIST Handwritten Digit Recognizer!");
        System.out.println("This is a fully connected feed forward neural network constructed in Java\nusing back propogation and stochastic gradient descent. The network can\nrecognize handwritten digits with near-human-level accuracy. Train a new network\nor open a previously saved network to test and analyze its capabilities.\n");
        System.out.print("Currently loaded network: ");
        // coloring file name
        if (currentLoadedNetwork == "NONE")
            System.out.print(RED + currentLoadedNetwork + RESET);
        else
            System.out.print(GREEN + currentLoadedNetwork + RESET);
        System.out.println("\n\nSelect an option:");
        System.out.println("[1] Train a new network");
        System.out.println("[2] Load a pre-trained network");
        // only show once a network is loaded in
        if (currentLoadedNetwork != "NONE") {
            System.out.println("[3] Display network accuracy on TRAINING data");
            System.out.println("[4] Display network accuracy on TESTING data");
            System.out.println("[5] Run network on TESTING data showing images and labels");
            System.out.println("[6] Display the misclassified TESTING images");
            System.out.println("[7] Save the network state to file");
        }
        System.out.println("[0] Exit\n");
    }

    // This function reads the training or testing data inputs and outputs and puts them into their arrays.
    public static void readMNISTData(File dataFile) throws FileNotFoundException {
        // get correct data size, inputs, and outputs based on input file
        int dataSize = 0;
        double[][] inputs = new double[0][0];
        double[][] outputs = new double[0][0];
        if (dataFile == TRAININGDATAFILE) {
            dataSize = TRAININGDATASIZE;
            inputs = trainingInputs;
            outputs = trainingOutputs;
        }
        else if (dataFile == TESTINGDATAFILE) {
            dataSize = TESTINGDATASIZE;
            inputs = testingInputs;
            outputs = testingOutputs;
        }

        // scanner to read data
        Scanner trainingScanner = new Scanner(dataFile);

        // reset digit totals
        for (int i=0; i<LAYERSIZES[2]; i++)
            totalOfEachDigit[i] = 0;

        // iterate thru all data
        for (int i=0; i<dataSize; i++) {
            String currLine = trainingScanner.nextLine();
            // store the image pixel data in inputs as a value 0-1 (scaled from 0-255)
            String[] currLineSplit = currLine.split(",");
            for (int j=1; j<currLineSplit.length; j++)
                inputs[i][j-1] = Double.parseDouble(currLineSplit[j])/255.0;
            // store the number the image displays in a one-hot vector in outputs
            for (int j=0; j<LAYERSIZES[2]; j++) {
                if (j == Character.getNumericValue(currLine.charAt(0)))
                    outputs[i][j] = 1.0;
                else
                    outputs[i][j] = 0.0;
            }
            // increment the total number of each handwritten digit
            totalOfEachDigit[Character.getNumericValue(currLine.charAt(0))]++;
        }
        trainingScanner.close();
    }

    // This function fills the inputted weight 2D array with initial random values -1 to 1.
    public static void createRandomInitialWeights(double[][] weightArray) {
        for (int i=0; i<weightArray.length; i++) {
            for (int j=0; j<weightArray[0].length; j++)
                weightArray[i][j] = 2.0*Math.random() - 1.0;
        }
    }

    // This function fills the inputted bias array with initial ranodm values -1 to 1.
    public static void createRandomInitialBiases(double[] biasArray) {
        for (int i=0; i<biasArray.length; i++)
            biasArray[i] = 2.0*Math.random() - 1.0;
    }

    // This function is the brains of training.
    // For a predefined number of epochs, the training data is split into predefined mini-batches.
    // Back propagation is performed on each input vector within a mini-batch, updating bias gradients
    // and weight gradients and their summations.
    // Finally, weight and bias values in the whole network are updated.
    public static void stochasticGradientDescent() {
        // repeat SGD through the whole input data a specified number of epochs
        for (int e=0; e<EPOCHS; e++) {
            // for every mini-batch
            for (int i=0; i<TRAININGDATASIZE/MINIBATCHSIZE; i++) {
                // reset summation arrays
                double[] layer1biasGradientsSummation = new double[LAYERSIZES[1]];
                double[] layer2biasGradientsSummation = new double[LAYERSIZES[2]];
                double[][] layer1weightGradientsSummation = new double[LAYERSIZES[1]][LAYERSIZES[0]];
                double[][] layer2weightGradientsSummation = new double[LAYERSIZES[2]][LAYERSIZES[1]];
                // for every input vector in a mini-batch
                for (int j=0; j<MINIBATCHSIZE; j++) {
                    // update activations and gradients with back propagation - pull the next random index to get the next input/output arrays
                    backPropagation(trainingInputs[trainingInputsRandomIndices[i*MINIBATCHSIZE + j]], trainingOutputs[trainingInputsRandomIndices[i*MINIBATCHSIZE + j]]);
                    // store the network's output
                    trainingNetworkOutputs[i*MINIBATCHSIZE + j] = layer2activations;
                    // update all bias and weight summation values
                    for (int k=0; k<LAYERSIZES[1]; k++) {
                        layer1biasGradientsSummation[k] += layer1biasGradients[k];
                        for (int l=0; l<LAYERSIZES[0]; l++)
                            layer1weightGradientsSummation[k][l] += layer1weightGradients[k][l];
                    }
                    for (int k=0; k<LAYERSIZES[2]; k++) {
                        layer2biasGradientsSummation[k] += layer2biasGradients[k];
                        for (int l=0; l<LAYERSIZES[1]; l++)
                            layer2weightGradientsSummation[k][l] += layer2weightGradients[k][l];
                    }
                }
                // now that mini-batch is finished, update weight and bias values based on learning rate, mini-batch size, and computed gradients
                for (int j=0; j<LAYERSIZES[1]; j++) {
                    layer1biases[j] = layer1biases[j] - (ETA / MINIBATCHSIZE) * layer1biasGradientsSummation[j];
                    for (int k=0; k<LAYERSIZES[0]; k++)
                        layer1weights[j][k] = layer1weights[j][k] - (ETA / MINIBATCHSIZE) * layer1weightGradientsSummation[j][k];
                }
                for (int j=0; j<LAYERSIZES[2]; j++) {
                    layer2biases[j] = layer2biases[j] - (ETA / MINIBATCHSIZE) * layer2biasGradientsSummation[j];
                    for (int k=0; k<LAYERSIZES[1]; k++)
                        layer2weights[j][k] = layer2weights[j][k] - (ETA / MINIBATCHSIZE) * layer2weightGradientsSummation[j][k];
                }
            }
        // calculate and display guessing statistics for this epoch
        generateOutputStatistics(e, 1);
        }
    }

    // This function calculates and displays the statistics for how accurate the network guessed outputs correctly.
    // It takes the current epoch as input to display this info.
    // It takes the menuCase as input to determine how to calculate statistics based on training or testing data.
    public static void generateOutputStatistics(int epoch, int menuCase) {
        // set vars based on menuCase
        int dataSize = 0;
        double[][] MNISTOutputs = new double[0][0];
        double[][] networkOutputs = new double[0][0];
        if (menuCase == 1 || menuCase == 3) {
            dataSize = TRAININGDATASIZE;
            MNISTOutputs = trainingOutputs;
            networkOutputs = trainingNetworkOutputs;
        }
        else if (menuCase == 4) {
            dataSize = TESTINGDATASIZE;
            MNISTOutputs = testingOutputs;
            networkOutputs = testingNetworkOutputs;
        }

        // start by setting all values to 0
        int[] correctOfEachDigit = new int[LAYERSIZES[2]];
        for (int i=0; i<LAYERSIZES[2]; i++)
            correctOfEachDigit[i] = 0;
        int totalCorrectDigits = 0;

        // increment handwritten digit if correct output - network matches mnist output
        for (int i=0; i<dataSize; i++) {
            // find what digit the network guessed
            double max = -1.0;
            int networkGuessedDigit = -1;
            for (int j=0; j<LAYERSIZES[2]; j++) {
                if (networkOutputs[i][j] > max) {
                    max = networkOutputs[i][j];
                    networkGuessedDigit = j;
                }
            }
            // find the correct output digit
            int MNISTCorrectDigit = -2;
            for (int j=0; j<LAYERSIZES[2]; j++) {
                if (menuCase == 1) {
                    if (MNISTOutputs[trainingInputsRandomIndices[i]][j] == 1.0) {
                        MNISTCorrectDigit = j;
                        break;
                    }
                }
                else if (menuCase == 3 || menuCase == 4) {
                    if (MNISTOutputs[i][j] == 1.0) {
                        MNISTCorrectDigit = j;
                        break;
                    }
                }
            }
            // compare the two, increment if correct
            if (networkGuessedDigit == MNISTCorrectDigit) {
                correctOfEachDigit[MNISTCorrectDigit]++;
                totalCorrectDigits++;
            }
        }

        // print the statistics output
        System.out.println("EPOCH " + (epoch+1) + "\n");
        String toPrint = "";
        for (int i=0; i<LAYERSIZES[2]; i++) {
            toPrint += "" + i + " = " + correctOfEachDigit[i] + "/" + totalOfEachDigit[i] + "\t";
            if (i == 5)
                toPrint += "\n";
        }
        toPrint += "Accuracy = " + totalCorrectDigits + "/" + dataSize + " = " + (Math.round((double)totalCorrectDigits/(double)dataSize*100000.0) / 1000.0) + "%";
        System.out.println(toPrint + "\n");
    }

    // This function takes in an input vector along with its desired output to perform back propagation.
    // First, the network is traversed feed forward to calculate activations at each node in each layer.
    // Next, the network is traversed backwards to calculate error values which are used to find bias gradients
    // and weight gradients at each node in each layer. 
    // These values are not returned but are stored in global variables.
    public static void backPropagation(double[] input, double[] desiredOutput) {
        // FEED FORWARD PASS - TRAVERSE NETWORK STARTING AT INPUT LAYER
        layer1activations = calcLayerActivations(layer1weights, input, layer1biases);
        layer2activations = calcLayerActivations(layer2weights, layer1activations, layer2biases);
        // BACKWARDS PASS - TRAVERSE NETWORK STARTING AT OUTPUT LAYER
        layer2biasGradients = calcErrorFinalLayer(layer2activations, desiredOutput);
        layer1biasGradients = calcErrorHiddenLayer(layer1activations, layer2biasGradients, layer2weights);
        for (int i=0; i<LAYERSIZES[2]; i++) {
            for (int j=0; j<LAYERSIZES[1]; j++)
                layer2weightGradients[i][j] = layer1activations[j] * layer2biasGradients[i];
        }
        for (int i=0; i<LAYERSIZES[1]; i++) {
            for (int j=0; j<LAYERSIZES[0]; j++)
                layer1weightGradients[i][j] = input[j] * layer1biasGradients[i];
        }
    }

    // This function takes in a layer's computed output and desired output and finds the error for each node in the layer
    // which is returned as a double array. This function is called by only the final layer of the network, the output layer.
    public static double[] calcErrorFinalLayer(double[] computedOutput, double[] desiredOutput) {
        double[] error = new double[computedOutput.length];
        for (int i=0; i<computedOutput.length; i++)
            error[i] = (computedOutput[i] - desiredOutput[i]) * computedOutput[i] * (1 - computedOutput[i]);
        return error;
    }

    // This function takes in a layer's computed output and the next layer's (this layer + 1) error values and weights
    // to return a double array of the error for each node in this layer. This function is called by only hidden layers,
    // all layers in the network except the input and output layers.
    public static double[] calcErrorHiddenLayer(double[] computedOutput, double[] nextLayerError, double[][] nextLayerWeights) {
        double[] error = new double[computedOutput.length];
        for (int i=0; i<computedOutput.length; i++) {
            double weightBiasSummation = 0.0;
            for (int j=0; j<nextLayerError.length; j++)
                weightBiasSummation += nextLayerWeights[j][i] * nextLayerError[j];
            error[i] = weightBiasSummation * computedOutput[i] * (1 - computedOutput[i]);
        }
        return error;
    }

    // This function takes in an entire layer's weights for each nueron, all inputs to the layer,
    // and all bias values and returns the array of nueron activations for this layer.
    public static double[] calcLayerActivations(double[][] layerWeights, double[] inputs, double[] layerBiases) {
        double[] neuronActivations = new double[layerBiases.length];
        // iterate through every neuron in the layer
        for (int i=0; i<layerBiases.length; i++)
            neuronActivations[i] = calcSigmoidFunction(layerWeights[i], inputs, layerBiases[i]);
        return neuronActivations;
    }

    // This function takes in an array of weights, an array of inputs, and a bias value
    // and returns y = sigmoid(z), the activation for a nueron.
    public static double calcSigmoidFunction(double[] weights, double[] inputs, double bias) {
        double z = calcDotSummation(weights, inputs, bias);
        return 1.0 / (1.0 +  Math.pow(Math.exp(1.0), -z));
    }

    // This function takes in an array of weights, an array of inputs, and a bias value
    // and returns z = the dot product of the weights and inputs plus the bias.
    public static double calcDotSummation(double[] weights, double[] inputs, double bias) {
        double dotProduct = calcDotProduct(weights, inputs);
        return dotProduct + bias;
    }

    // This function takes in 2 double arrays and returns the dot product of them.
    public static double calcDotProduct(double[] arr1, double[] arr2) {
        double dotProduct = 0.0;
        for (int i=0; i<arr1.length; i++)
            dotProduct += arr1[i]*arr2[i];
        return dotProduct;
    }
}