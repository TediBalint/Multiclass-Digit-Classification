namespace Multiclass_Digit_Classification
{
	internal class Program
	{
		static void Main(string[] args)
		{
			//NeuralNetwork neuralNetwork = new NeuralNetwork(0.01, Statics.TRAININGFOLDERNAME, 200);
			//NeuralNetwork neuralNetwork = new NeuralNetwork(0.000001, Statics.TESTINGFOLDERNAME, 1000, "weights1.txt");
			//neuralNetwork.Train(10000, 64, false, true);
			//neuralNetwork.Test(10000);

			ShowCase();


		}
		private static void ShowCase()
		{
			NeuralNetwork neuralNetwork = new NeuralNetwork(0.000001, Statics.TESTINGFOLDERNAME, 1000, "weights1.txt");
			while (true) 
			{
				Image randImage = neuralNetwork.GetRandomImage();
				randImage.Show();
                Console.WriteLine($"Prediction: {neuralNetwork.GetOutput(randImage.ImageData)}");
                Console.WriteLine($"Actual label: {randImage.Label}");
				Console.ReadKey();

			}
		}
	}
}
