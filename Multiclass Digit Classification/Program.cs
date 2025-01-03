namespace Multiclass_Digit_Classification
{
	internal class Program
	{
		static void Main(string[] args)
		{
			ShowCase();
		}
		private static void ShowCase()
		{
			NeuralNetwork neuralNetwork = new NeuralNetwork(0.000001, Statics.TESTINGFOLDERNAME, 1000, Statics.LOWESTWEIGHSTFILE);
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
