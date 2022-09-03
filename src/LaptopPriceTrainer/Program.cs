using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Collections;
using Microsoft.ML;

namespace LaptopPriceTrainer
{
    class Program
    {
        static string dataPath = "laptoppricesnew.csv";
        static void Main(string[] args)
        {
            var mlContext = new MLContext(seed: 1);

            // Load the Data
            var data = mlContext.Data.LoadFromTextFile<DataSchema>(dataPath, hasHeader: true, separatorChar: ',');
            
            // Split the Dataset
            var testTrainData = mlContext.Data.TrainTestSplit(data, testFraction:0.2);

            // var dataProcessingPipeline = mlContext.Transforms.Categorical.OneHotHashEncoding("CPU")
            // .Append(mlContext.Transforms.Categorical.OneHotHashEncoding("GPU"))
            // .Append(mlContext.Transforms.Categorical.OneHotHashEncoding("RAMType"))
            // .Append(mlContext.Transforms.Concatenate("Features", "CPU", "GPU", "RAMType",
            // "GHz", "RAM", "Storage", "SSD"));

            var dataProcessingPipeline = mlContext.Transforms.Categorical.OneHotHashEncoding("CPU")
            .Append(mlContext.Transforms.Categorical.OneHotHashEncoding("GPU"))
            .Append(mlContext.Transforms.Categorical.OneHotHashEncoding("RAMType"))
            .Append(mlContext.Transforms.Concatenate("Features", "CPU", "GPU", "RAMType",
            "GHz", "RAM", "Storage", "SSD"));




        }
    }
}
