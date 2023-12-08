#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>

// Function Declarations
char **addWord(char **list, int *size, char *word);
int isExist(char **list, int size, char *word);
void readData(int ***hotVectors, int **labels, int *wordsSize, int *lines);
double mean_squared_error(int y_true[], double y_pred[], int n);
double mse_derivative(double y_true, double y_pred);
double tanh_function(double x);
double tanh_derivative(double x);
void writeCSV(const char *filename, double *times, double *losses, int epochs);
void writeWeightsCSV(const char *filename, double **weights, int epochs, int wordsSize);
void shuffleData(int ***hotVectors, int **labels, int lines);
void ADAM(int **hotVectors, int *labels, int wordsSize, int lines, int fileIndex, double initialValue, int epochs, double learning_rate, int batch_size);
void GD(int **hotVectors, int *labels, int wordsSize, int lines,int fileIndex,double initialValue,int epochs,double learning_rate);
void SGD(int **hotVectors, int *labels, int wordsSize, int lines,int fileIndex,double initialValue,int epochs,double learning_rate,int batch_size);

// Main Function
int main() {
    int **hotVectors = NULL;
    int *labels = NULL;
    int wordsSize = 0;
    int lines = 0;
    int epochs = 200;  // Number of epochs  
    double learning_rate = 0.001;//learning rate
    int batch_size = 64;//batch size
    readData(&hotVectors, &labels, &wordsSize, &lines);//read data from csv file
    shuffleData(&hotVectors, &labels, lines);//shuffle data

    double initialValues[5]={0.001,0.0,0.1,0.5,1.0};//initial values for weights
    for(int i=0;i<1;i++){
        
        GD(hotVectors, labels, wordsSize, lines,i+1,initialValues[i],epochs,learning_rate);
        SGD(hotVectors, labels, wordsSize, lines,i+1,initialValues[i],epochs,learning_rate,batch_size);    
        ADAM(hotVectors, labels, wordsSize, lines,i+1,initialValues[i],epochs,learning_rate,batch_size);
    }

    // Freeing hotVectors and labels
    for (int i = 0; i < lines; i++) {
        free(hotVectors[i]);
    }
    free(hotVectors);
    free(labels);

    return 0;
}


// SGD Optimizer Function
//@param hotVectors: 2D array of hot vectors
//@param labels: 1D array of labels
//@param wordsSize: number of words
//@param lines: number of lines
//@param fileIndex: index of file
//@param initialValue: initial value of weights
void SGD(int **hotVectors, int *labels, int wordsSize, int lines,int fileIndex,double initialValue,int epochs,double learning_rate,int batch_size) {
    srand(time(NULL));
    
    double *weights = calloc(wordsSize, sizeof(double));
    int train_size = (int)(lines * 0.8);  
    


    for(int i = 0; i < wordsSize; i++){//initialize weights  
        weights[i] = initialValue;
    }

    double **epochWeights = malloc(epochs * sizeof(double *));//array to store weights for each epoch
    for (int i = 0; i < epochs; i++) {
        epochWeights[i] = malloc(wordsSize * sizeof(double));
    }

    double total_loss = 0.0;
    int correct_predictions = 0;

    double *times = malloc(epochs * sizeof(double));
    double *losses = malloc(epochs * sizeof(double));


    struct timeval start, end;

    gettimeofday(&start, NULL);//start timer

    // Train stage
    for (int epoch = 0; epoch < epochs; epoch++) {//for each epoch
        total_loss = 0.0;
        correct_predictions = 0;


        for (int i = 0; i < batch_size; i++) {//istead of using all data for each epoch, we use batch_size data for each epoch
            
            int index = rand() % train_size;//randomly select data from train data
            double dot_product = 0.0;
            for (int j = 0; j < wordsSize; j++) {
                dot_product += hotVectors[index][j] * weights[j];//w*xi
            }

            double prediction = tanh_function(dot_product); //tanh(w*xi)           
            double error = mse_derivative(labels[index], prediction);//2*(y-tanh(w*xi))

            for (int j = 0; j < wordsSize; j++) {
                weights[j] -= learning_rate * error * tanh_derivative(prediction) * hotVectors[index][j];//w=w-2*learning_rate*(y-tanh(w*xi))*tanh'(w*xi)*xi
            }

            total_loss += mean_squared_error(&labels[index], &prediction, 1);//total loss function for each epoch
            if ((prediction >= 0 && labels[index] == 1) || (prediction < 0 && labels[index] == -1)) {//count correct predictions for each epoch to calculate accuracy
                correct_predictions++;
            }
        }

        gettimeofday(&end, NULL);
        double time_taken = ((end.tv_sec - start.tv_sec) * 1000.0f + (end.tv_usec - start.tv_usec) / 1000.0f) / 1000.0f;//calculate time for each epoch
        times[epoch] = time_taken;//store time for each epoch
        losses[epoch] = total_loss / train_size;//store loss for each epoch

        // Store the weights for this epoch
        for (int j = 0; j < wordsSize; j++) {
            epochWeights[epoch][j] = weights[j];//store weights for each epoch
        }

        double accuracy = (double)correct_predictions / (batch_size);//calculate accuracy for each epoch
        printf("Epoch %d: Loss = %f, Accuracy = %f\n", epoch, losses[epoch], accuracy);//print loss and accuracy for each epoch
    }

    char filename[100];
    sprintf(filename, "C:\\Users\\BTK\\Desktop\\diferansiyel\\SGD\\loss_time_epocks\\gd_performance%d.csv", fileIndex);//write losses,Times and epochs to csv files
    writeCSV(filename, times, losses, epochs);

    // Write weights to CSV
    sprintf(filename, "C:\\Users\\BTK\\Desktop\\diferansiyel\\SGD\\weigts\\weights_per_epoch%d.csv", fileIndex);//write weights to csv files
    writeWeightsCSV(filename, epochWeights, epochs, wordsSize);

    
    
    total_loss = 0.0;
    correct_predictions = 0;

    // Test stage
    for(int i=train_size+1;i<lines;i++){
        
        double dot_product = 0.0;
        for (int j = 0; j < wordsSize; j++) {
            dot_product += hotVectors[i][j] * weights[j];  //w*xi
        }

        double prediction = tanh_function(dot_product);//tanh(w*xi)
        total_loss += mean_squared_error(&labels[i], &prediction, 1);//total loss function for each epoch

        if ((prediction >= 0.0 && labels[i] == 1) || (prediction < 0.0 && labels[i] == -1)) {//count correct predictions for each epoch to calculate accuracy
            
            correct_predictions+=1;
        }
        
        
    }
    double avg_loss = total_loss / (lines-train_size);//calculate average loss
    double accuracy = (double)correct_predictions / (lines-train_size);//calculate accuracy for Test stage
    printf("Test: Loss = %f, Accuracy = %f\n", avg_loss, accuracy);

    printf("\n############################################\n");
    printf("\n############################################\n");
    printf("\n############################################\n");
    printf("\n############################################\n");
    printf("\n############################################\n");

    free(times);
    free(losses);
    free(weights);

    // Free epochWeights
    for (int i = 0; i < epochs; i++) {
        free(epochWeights[i]);
    }
    free(epochWeights);
}


// ADAM Optimizer Function
//@param hotVectors: 2D array of hot vectors
//@param labels: 1D array of labels
//@param wordsSize: number of words
//@param lines: number of lines
//@param fileIndex: index of file
//@param initialValue: initial value of weights
void ADAM(int **hotVectors, int *labels, int wordsSize, int lines, int fileIndex, double initialValue,int epochs,double learning_rate,int batch_size) {
    
    srand(time(NULL));
    double *weights = calloc(wordsSize, sizeof(double));
    int train_size = (int)(lines * 0.8);


    // ADAM-specific initializations
    double beta1 = 0.9;//this used for momentum
    double beta2 = 0.999;//this used for velocity
    double epsilon = 1e-8;//this used for numerical stability
    double *m = calloc(wordsSize, sizeof(double));//momentum vector
    double *v = calloc(wordsSize, sizeof(double));//velocity vector

    // Initialize weights
    for(int i = 0; i < wordsSize; i++){
        weights[i] = initialValue;
    }
    

    double **epochWeights = malloc(epochs * sizeof(double *));//array to store weights for each epoch
    for (int i = 0; i < epochs; i++) {
        epochWeights[i] = malloc(wordsSize * sizeof(double));
    }

    double total_loss = 0.0;
    int correct_predictions = 0;

    double *times = malloc(epochs * sizeof(double));
    double *losses = malloc(epochs * sizeof(double));

    struct timeval start, end;

    gettimeofday(&start, NULL);//start timer

    // ADAM Training Stage
    for (int epoch = 0; epoch < epochs; epoch++) {//for each epoch
        total_loss = 0.0;
        correct_predictions = 0;

        for (int i = 0; i < batch_size; i++) {//for each training data
            
            int index = rand() % train_size;//randomly select data from train data    
            
            double dot_product = 0.0;

            for (int j = 0; j < wordsSize; j++) {//calculate dot product==>w.xi
                dot_product += hotVectors[index][j] * weights[j];
            }

            double prediction = tanh_function(dot_product);//calculate tanh(w.xi)
            double error = mse_derivative(labels[index], prediction);//calculate 2*(y-tanh(w.xi))

            for (int j = 0; j < wordsSize; j++) {
                double gradient = error * tanh_derivative(prediction) * hotVectors[index][j];//calculate 2*(y-tanh(w.xi))*tanh'(w*xi)*xi

                // ADAM Update
                m[j] = beta1 * m[j] + (1 - beta1) * gradient;//momentum vector m=beta1*m+(1-beta1)*gradient
                v[j] = beta2 * v[j] + (1 - beta2) * gradient * gradient;//velocity vector v=beta2*v+(1-beta2)*gradient^2
                double m_hat = m[j] ;
                double v_hat = v[j] ;

                
                weights[j] -= learning_rate * m_hat / (sqrt(v_hat + epsilon));//w=w-learning_rate*m_hat/(sqrt(v_hat+epsilon))
            }

            total_loss += mean_squared_error(&labels[index], &prediction, 1);//total loss function for each epoch
            if ((prediction >= 0 && labels[index] == 1) || (prediction < 0 && labels[index] == -1)) {//count correct predictions for each epoch to calculate accuracy
                correct_predictions++;
            }
        }

        gettimeofday(&end, NULL);
        double time_taken = ((end.tv_sec - start.tv_sec) * 1000.0f + (end.tv_usec - start.tv_usec) / 1000.0f) / 1000.0f;//calculate time for each epoch
        times[epoch] = time_taken;
        losses[epoch] = total_loss / train_size;

        // Store weights for this epoch
        for (int j = 0; j < wordsSize; j++) {
            epochWeights[epoch][j] = weights[j];
        }

        double accuracy = (double)correct_predictions / batch_size;//calculate accuracy for each epoch
        printf("Epoch %d: Loss = %f, Accuracy = %f\n", epoch, losses[epoch], accuracy);//print loss and accuracy for each epoch
    }


    char filename[100];
    sprintf(filename, "C:\\Users\\BTK\\Desktop\\diferansiyel\\ADAM\\loss_time_epocks\\gd_performance%d.csv", fileIndex);//write losses,time and epochs to csv files 
    writeCSV(filename, times, losses, epochs);

    // Write weights to CSV
    sprintf(filename, "C:\\Users\\BTK\\Desktop\\diferansiyel\\ADAM\\weigts\\weights_per_epoch%d.csv", fileIndex);//write weights to csv files
    writeWeightsCSV(filename, epochWeights, epochs, wordsSize);



    // Test stage
    total_loss = 0.0;
    correct_predictions = 0;
    for(int i=train_size+1;i<lines;i++){
        
        double dot_product = 0.0;
        for (int j = 0; j < wordsSize; j++) {
            dot_product += hotVectors[i][j] * weights[j];  //w*xi
        }

        double prediction = tanh_function(dot_product);//tanh(w*xi)
        
        total_loss += mean_squared_error(&labels[i], &prediction, 1);//total loss function for each epoch

        if ((prediction >= 0.0 && labels[i] == 1) || (prediction < 0.0 && labels[i] == -1)) {//count correct predictions for each epoch to calculate accuracy
            
            correct_predictions+=1;
        }
        
        
    }
    double avg_loss = total_loss / (lines-train_size);//calculate average loss
    double accuracy = (double)correct_predictions / (lines-train_size);//calculate accuracy for Test stage
    printf("Test: Loss = %f, Accuracy = %f\n", avg_loss, accuracy);//print loss and accuracy for Test stage

    printf("\n############################################\n");
    printf("\n############################################\n");
    printf("\n############################################\n");
    printf("\n############################################\n");
    printf("\n############################################\n");

    // Free allocated memory
    free(m);
    free(v);
    free(times);
    free(losses);
    free(weights);

    // Free epochWeights
    for (int i = 0; i < epochs; i++) {
        free(epochWeights[i]);
    }
    free(epochWeights);
}


// GD Optimizer Function
//@param hotVectors: 2D array of hot vectors
//@param labels: 1D array of labels
//@param wordsSize: number of words
//@param lines: number of lines
//@param fileIndex: index of file
void GD(int **hotVectors, int *labels, int wordsSize, int lines,int fileIndex,double initialValue,int epochs,double learning_rate) {
    srand(time(NULL));
    
    double *weights = calloc(wordsSize, sizeof(double));  
    int train_size = (int)(lines * 0.8);  

    for(int i = 0; i < wordsSize; i++){//initialize weights  
        weights[i] = initialValue;
    }

    double **epochWeights = malloc(epochs * sizeof(double *));//array to store weights for each epoch
    for (int i = 0; i < epochs; i++) {
        epochWeights[i] = malloc(wordsSize * sizeof(double));
    }

    double total_loss = 0.0;
    int correct_predictions = 0;

    double *times = malloc(epochs * sizeof(double));
    double *losses = malloc(epochs * sizeof(double));


    struct timeval start, end;

    gettimeofday(&start, NULL);//start timer

    // Train stage
    for (int epoch = 0; epoch < epochs; epoch++) {//for each epoch
        total_loss = 0.0;
        correct_predictions = 0;

        for (int i = 0; i < train_size; i++) {
            double dot_product = 0.0;
            for (int j = 0; j < wordsSize; j++) {
                dot_product += hotVectors[i][j] * weights[j];//w*xi
            }

            double prediction = tanh_function(dot_product); //tanh(w*xi)           
            double error = mse_derivative(labels[i], prediction);//2*(y-tanh(w*xi))

            for (int j = 0; j < wordsSize; j++) {
                weights[j] -= (learning_rate * error * tanh_derivative(prediction) * hotVectors[i][j]);//w=w-2*learning_rate*(y-tanh(w*xi))*tanh'(w*xi)*xi
            }

            total_loss += mean_squared_error(&labels[i], &prediction, 1);//total loss function for each epoch
            if ((prediction >= 0 && labels[i] == 1) || (prediction < 0 && labels[i] == -1)) {//count correct predictions for each epoch to calculate accuracy
                correct_predictions++;
            }
        }

        gettimeofday(&end, NULL);
        double time_taken = ((end.tv_sec - start.tv_sec) * 1000.0f + (end.tv_usec - start.tv_usec) / 1000.0f) / 1000.0f;//calculate time for each epoch
        times[epoch] = time_taken;//store time for each epoch
        losses[epoch] = total_loss / train_size;//store loss for each epoch

        // Store the weights for this epoch
        for (int j = 0; j < wordsSize; j++) {
            epochWeights[epoch][j] = weights[j];//store weights for each epoch
        }

        double accuracy = (double)correct_predictions / train_size;//calculate accuracy for each epoch
        printf("Epoch %d: Loss = %f, Accuracy = %f\n", epoch, losses[epoch], accuracy);//print loss and accuracy for each epoch
    }

    char filename[100];
    sprintf(filename, "C:\\Users\\BTK\\Desktop\\diferansiyel\\GD\\loss_time_epocks\\gd_performance%d.csv", fileIndex);//write losses,time and epochs to csv files 
    writeCSV(filename, times, losses, epochs);

    // Write weights to CSV
    sprintf(filename, "C:\\Users\\BTK\\Desktop\\diferansiyel\\GD\\weigts\\weights_per_epoch%d.csv", fileIndex);//write weights to csv files
    writeWeightsCSV(filename, epochWeights, epochs, wordsSize);

    

    total_loss = 0.0;
    correct_predictions = 0;

    // Test stage
    for(int i=train_size+1;i<lines;i++){
        
        double dot_product = 0.0;
        for (int j = 0; j < wordsSize; j++) {
            dot_product += hotVectors[i][j] * weights[j];  //w*xi
        }

        double prediction = tanh_function(dot_product);//tanh(w*xi)
        
        total_loss += mean_squared_error(&labels[i], &prediction, 1);//total loss function for each epoch

        if ((prediction >= 0.0 && labels[i] == 1) || (prediction < 0.0 && labels[i] == -1)) {//count correct predictions for each epoch to calculate accuracy
            
            correct_predictions+=1;
        }
        
        
    }
    double avg_loss = total_loss / (lines-train_size);//calculate average loss
    double accuracy = (double)correct_predictions / (lines-train_size);//calculate accuracy for Test stage
    printf("Test: Loss = %f, Accuracy = %f\n", avg_loss, accuracy);//print loss and accuracy for Test stage

    printf("\n############################################\n");
    printf("\n############################################\n");
    printf("\n############################################\n");
    printf("\n############################################\n");
    printf("\n############################################\n");

    free(times);
    free(losses);
    free(weights);

    // Free epochWeights
    for (int i = 0; i < epochs; i++) {
        free(epochWeights[i]);
    }
    free(epochWeights);
}

// Shuffle Data
//@param hotVectors: 2D array of hot vectors
//@param labels: 1D array of labels
//@param lines: number of lines
void shuffleData(int ***hotVectors, int **labels, int lines) {
    for (int i = 0; i < lines; i++) {
        int j = rand() % lines;

        // Swap hotVectors[i] and hotVectors[j]
        int *tempVector = (*hotVectors)[i];
        (*hotVectors)[i] = (*hotVectors)[j];
        (*hotVectors)[j] = tempVector;

        // Swap labels[i] and labels[j]
        int tempLabel = (*labels)[i];
        (*labels)[i] = (*labels)[j];
        (*labels)[j] = tempLabel;
    }
}


//mean squared error function==>(y_pred-y_true)^2
//@param y_true: true value
//@param y_pred: predicted value
//@param n: number of elements
//@return: mean squared error

double mean_squared_error(int y_true[], double y_pred[], int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        double diff = (double)y_true[i] - y_pred[i];
        sum += diff * diff;
    }
    return sum / (double)n;
}


//mean squared error derivative function==>2*(y_pred-y_true)
//@param y_true: true value
//@param y_pred: predicted value
//@return: mean squared error

double mse_derivative(double y_true, double y_pred) {
    return 2 * (y_pred - y_true);
}

//tanh function
//@param x: input value
//@return: tanh value
double tanh_function(double x) {
    return tanh(x);
}

//tanh derivative function
//@param x: input value
//@return: tanh derivative value

double tanh_derivative(double x) {
    double tanh_x = tanh(x);
    return 1.0 - tanh_x * tanh_x;
}

//check if word is exist in list
//@param list: list of words
//@param size: size of list
//@param word: word to check
//@return: 1 if exist, 0 if not exist
int isExist(char **list, int size, char *word) {
    for (int i = 0; i < size; i++) {
        if (strcmp(list[i], word) == 0) {
            return 1;
        }
    }
    return 0;
}

//add word to list
//@param list: list of words
//@param size: size of list
//@param word: word to add  
//@return: list of words
char **addWord(char **list, int *size, char *word) {
    list = realloc(list, (*size + 1) * sizeof(char *));
    list[*size] = strdup(word);
    (*size)++;
    return list;
}

//read data from csv file
//@param hotVectors: 2D array of hot vectors
//@param labels: 1D array of labels
//@param wordsSize: number of words
//@param lines: number of lines
void readData(int ***hotVectors, int **labels, int *wordsSize, int *lines) {
    FILE *fp;
    fp = fopen("data_set_1000.csv", "r");

    if (fp == NULL) {
        printf("Error in opening file\n");
        exit(1);
    }

    char buf[10000];
    char **words = NULL;
    *wordsSize = 0;
    *lines = 0;
    int labelsSize = 0;

    while (fgets(buf, 10000, fp) != NULL) {//read words and build the list of unique words
        char *token = strtok(buf, ", \n");

        while (token != NULL) {
            if (!isExist(words, *wordsSize, token)) {
                words = addWord(words, wordsSize, token);
            }
            token = strtok(NULL, ", \n");
        }
    }

    // Rewind the file to start reading data again
    fseek(fp, 0, SEEK_SET);


    // Read data and build hot vectors and labels
    while (fgets(buf, 10000, fp) != NULL) {//loop for each line
        char *token = strtok(buf, ", \n");
        int *hotVector = calloc(*wordsSize, sizeof(int));

        while (token != NULL) {//loop for each word in line
            char *nextToken = strtok(NULL, ", \n");
            if (nextToken != NULL) {
                int index = -1;
                for (int i = 0; i < *wordsSize; i++) {
                    if (strcmp(words[i], token) == 0) {//check if word is exist in dictionary
                        index = i;
                        break;
                    }
                }
                if (index != -1) {//if word is exist in dictioary, set hot vector to 1
                    hotVector[index] = 1;
                }
                token = nextToken;
            } else {
                *labels = realloc(*labels, (labelsSize + 1) * sizeof(int));//set label
                (*labels)[labelsSize++] = atoi(token);
                break;
            }
        }

        *hotVectors = realloc(*hotVectors, (*lines + 1) * sizeof(int *));//add hot vector to hot vectors array
        (*hotVectors)[(*lines)++] = hotVector;//add label to labels array
    }

    // Free words
    for (int i = 0; i < *wordsSize; i++) {
        free(words[i]);
    }
    free(words);

    fclose(fp);
}

//write loss,time and epochs to csv file
//@param filename: name of file
//@param times: array of times
//@param losses: array of losses
//@param epochs: number of epochs
void writeCSV(const char *filename, double *times, double *losses, int epochs) {

    FILE *fp = fopen(filename, "w");
    if (fp == NULL) {
        printf("Error opening file!\n");
        return;
    }

    fprintf(fp, "Epoch,Time,Loss\n");
    for (int i = 0; i < epochs; i++) {
        fprintf(fp, "%d,%f,%f\n", i, times[i], losses[i]);
    }

    fclose(fp);
}

//write weights to csv file
//@param filename: name of file
//@param weights: array of weights
//@param epochs: number of epochs
//@param wordsSize: number of words

void writeWeightsCSV(const char *filename, double **weights, int epochs, int wordsSize) {
    FILE *fp = fopen(filename, "w");
    if (fp == NULL) {
        printf("Error opening file!\n");
        return;
    }

    // Write header
    fprintf(fp, "Epoch");
    for (int i = 0; i < wordsSize; i++) {
        fprintf(fp, ",Weight%d", i);
    }
    fprintf(fp, "\n");

    // Write data
    for (int epoch = 0; epoch < epochs; epoch++) {
        fprintf(fp, "%d", epoch);
        for (int j = 0; j < wordsSize; j++) {
            fprintf(fp, ",%f", weights[epoch][j]);
        }
        fprintf(fp, "\n");
    }

    fclose(fp);
}
