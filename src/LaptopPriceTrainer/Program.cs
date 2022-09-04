using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Collections;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace LaptopPriceTrainer
{
    class Program
    {
        // Path to the dataset
        static string dataPath = @"/home/daniel/Desktop/programming/pythondatascience/datascience/c_sharp/dataset/laptoppricesnew.csv";
        static void Main(string[] args)
        {
            var mlContext = new MLContext(seed: 1);

            // Load the Data
            var data = mlContext.Data.LoadFromTextFile<DataSchema>(
                path: dataPath,
                hasHeader: true,
                separatorChar: ','
            );

            // Split the Dataset
            var testTrainData = mlContext.Data.TrainTestSplit(data, testFraction:0.2);

            var dataProcessingPipeline = mlContext.Transforms.Categorical.OneHotEncoding(nameof(DataSchema.CPU))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(nameof(DataSchema.GPU)))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(nameof(DataSchema.RAMType)))
                .Append(mlContext.Transforms.Concatenate("Features", nameof(DataSchema.CPU), 
                nameof(DataSchema.GPU), nameof(DataSchema.RAMType), nameof(DataSchema.GHz),
                nameof(DataSchema.RAM), nameof(DataSchema.Weight), nameof(DataSchema.Screen), nameof(DataSchema.Storage),
                nameof(DataSchema.SSD)));

            Console.WriteLine("Start training model");
            var startTime = DateTime.Now;
            var fastTreettrainingPipeline = dataProcessingPipeline
                .Append(mlContext.Regression.Trainers.FastTreeTweedie(
                    labelColumnName: nameof(DataSchema.Price),
                    featureColumnName: "Features",
                    numberOfLeaves: 30,
                    learningRate: 0.24,
                    numberOfTrees: 60));
            

            var fastTtrainingPipeline = dataProcessingPipeline
                .Append(mlContext.Regression.Trainers.FastTree(
                    labelColumnName: nameof(DataSchema.Price),
                    featureColumnName: "Features",
                    numberOfLeaves: 30,
                    learningRate: 0.24,
                    numberOfTrees: 60));

            var lightGtrainingPipeline = dataProcessingPipeline
                .Append(mlContext.Regression.Trainers.LightGbm(
                    labelColumnName: nameof(DataSchema.Price),
                    featureColumnName: "Features",
                    numberOfLeaves: 20,
                    learningRate: 0.25,
                    numberOfIterations: 50
                ));
                
            Console.WriteLine($"Model training finished in {(DateTime.Now - startTime).TotalSeconds} seconds");

            var fastTreettrainedModel = fastTreettrainingPipeline.Fit(testTrainData.TrainSet);
            var fastTtrainedModel = fastTtrainingPipeline.Fit(testTrainData.TrainSet);
            var lightGtrainedModel = lightGtrainingPipeline.Fit(testTrainData.TrainSet);

            var fastTreetpreds = fastTreettrainedModel.Transform(testTrainData.TestSet);
            var ftpreds = fastTtrainedModel.Transform(testTrainData.TestSet);
            var lgpreds = lightGtrainedModel.Transform(testTrainData.TestSet);

            var fastTreetmetrics = mlContext.Regression.Evaluate(fastTreetpreds, labelColumnName: nameof(DataSchema.Price));
            var ftmetrics = mlContext.Regression.Evaluate(ftpreds, labelColumnName: nameof(DataSchema.Price));
            var lgmetrics = mlContext.Regression.Evaluate(lgpreds, labelColumnName: nameof(DataSchema.Price));

            Console.WriteLine($"\nFast Forest RSquared Score: {fastTreetmetrics.RSquared:0.#####}");
            Console.WriteLine($"Fast Forest RMSE Score: {fastTreetmetrics.RootMeanSquaredError:0.#####}\n");
            Console.WriteLine($"Fast Tree RSquared Score: {ftmetrics.RSquared:0.#####}");
            Console.WriteLine($"Fast Tree RMSE Score: {ftmetrics.RootMeanSquaredError:0.#####}\n");
            Console.WriteLine($"LightGbm RSquared Score: {lgmetrics.RSquared:0.#####}");
            Console.WriteLine($"LightGbm RMSE Score: {lgmetrics.RootMeanSquaredError:0.#####}\n");
        }
    }
}
