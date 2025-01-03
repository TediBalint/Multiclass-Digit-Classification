using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data.Common;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Multiclass_Digit_Classification
{
	public class NeuralNetwork
	{
		private readonly List<double> _input_hidden_weights;
		private readonly List<double> _hidden_output_weights;
		private readonly List<double> _hidden_biases;
		private readonly List<double> _output_biases;

		private double _lr;

		private DataLoader _loader;

		private int run_counter = 0;
		public NeuralNetwork(double lr, string inputFolder, int datasize)
		{
			_input_hidden_weights = new List<double>();
			_hidden_output_weights = new List<double>();
			_hidden_biases = new List<double>();
			_output_biases = new List<double>();

			_lr = lr;
			_loader = new DataLoader(inputFolder, datasize);
			randomizeWeightsBiases();
		}
		public NeuralNetwork(double lr, string inputFolder, int datasize, string saveFilePath)
		{
			
			using (StreamReader sr = new StreamReader(saveFilePath))
			{
				_input_hidden_weights = sr.ReadLine().Split(';').Select(double.Parse).ToList();
				_hidden_output_weights = sr.ReadLine().Split(';').Select(double.Parse).ToList();
				_hidden_biases = sr.ReadLine().Split(';').Select(double.Parse).ToList();
				_output_biases = sr.ReadLine().Split(';').Select(double.Parse).ToList();
			}
			_lr = lr;
			_loader = new DataLoader(inputFolder, datasize);
		}
		
		public void Train(int baches, int bachsize, bool save = false, bool debug = false)
		{
			for (int i = 0; i < baches; i++)
			{
				run_counter++;
				if (debug && run_counter % 100 == 0) Console.Write($"Bach {i + 1}.\t");
				runBach(bachsize, debug);
			}
			if (save) Save();
		}
		public void Test(int amount, int shuffle_frequency = 100)
		{
			Console.WriteLine("STARTING TEST");
			Dataset dataset = _loader.GetDataset();
			int correct = 0;
			for(int i = 0;i < amount; i++)
			{
				if (i % shuffle_frequency == 0) dataset.Shuffle();
				Image image = dataset.NextImage;
				int myPrediction = GetOutput(image.ImageData);
				if(myPrediction == image.Label) correct++;
			}
			Console.WriteLine($"TEST FINISHED: {correct}/{amount}");
		}
		public int GetOutput(List<double> input)
		{
			List<double> output = computeOutputLayerValues(_hidden_output_weights, computeHiddenLayerValues(_input_hidden_weights, input, _hidden_biases), _output_biases);
			int outIndex = 0;
			for (int i = 1;i < output.Count; i++)
			{
				if (output[i] > output[outIndex]) outIndex = i;
			}
			return outIndex;
		}
		public Image GetRandomImage()
		{
			Dataset dataset = _loader.GetDataset();
			dataset.Shuffle();
			return dataset.NextImage;
		}
		private void runBach(int bachsize, bool debug)
		{
			int n = _hidden_output_weights.Count / _output_biases.Count;
			int m = _input_hidden_weights.Count / _hidden_biases.Count;

			Dataset dataset = _loader.GetDataset();
			dataset.Shuffle();

			List<double> hidden_output_weight_gradient_avgs = Enumerable.Repeat(0.0, _hidden_output_weights.Count).ToList();
			List<double> input_hidden_weight_gradient_avgs = Enumerable.Repeat(0.0, _input_hidden_weights.Count).ToList();
			List<double> output_bias_avgs = Enumerable.Repeat(0.0, _output_biases.Count).ToList();
			List<double> hidden_bias_avgs = Enumerable.Repeat(0.0, _hidden_biases.Count).ToList();
			double loss_avg = 0;


			for (int i = 0; i < bachsize; i++)
			{
				Image curr_image = dataset.NextImage;

				List<int> true_table = Enumerable.Repeat(0, _output_biases.Count).ToList();

				int correct_anwser = curr_image.Label;
				true_table[correct_anwser] = 1;

				List<double> input_values = curr_image.ImageData;
				List<double> rawHiddenLayerValues = computeHiddenLayerValuesRaw(_input_hidden_weights, input_values, _hidden_biases);
				List<double> activatedHiddenLayerValues = computeHiddenLayerValues(rawHiddenLayerValues);
				List<double> rawOutputValues = computeOutputLayerRawValues(_hidden_output_weights, activatedHiddenLayerValues, _output_biases);
				List<double> activatedOutputLayerValues = computeOutputLayerValues(rawOutputValues);

				double correct_prediction = activatedOutputLayerValues[correct_anwser];
					
				for (int j = 0; j < output_bias_avgs.Count; j++) // j az output layerhez tartozo
				{
					int true_label = getKroneckerDelta(j, correct_anwser);
					double prediction = activatedOutputLayerValues[j];

					double output_to_bias = getOutputGradientToBias(prediction, correct_prediction, true_label);
					output_bias_avgs[j] += output_to_bias;

					for (int k = 0; k < n; k++) // k hidden layerhez tartozo
					{
						double hidden_activation_output = activatedHiddenLayerValues[k];
						hidden_output_weight_gradient_avgs[k + j * n] += getOutputGradientToWeight(output_to_bias, hidden_activation_output);
					}
				}

				for (int j = 0; j < hidden_bias_avgs.Count; j++)
				{
					double contribution = getHiddenLayerContributionToOutput(j, correct_anwser, activatedOutputLayerValues);
					double reluGradient = getReLUGradient(rawHiddenLayerValues[j]);

					double hidden_bias = getHiddenGradientToBias(contribution, reluGradient);
					hidden_bias_avgs[j] += hidden_bias;

					for (int k = 0; k < m; k++)
					{
						input_hidden_weight_gradient_avgs[k + j * m] += getHiddenGradientToWeight(hidden_bias, input_values[k]);
					}
				}

				loss_avg += getLoss(activatedOutputLayerValues, true_table);

			}

			input_hidden_weight_gradient_avgs = input_hidden_weight_gradient_avgs.Select(x => x /= bachsize).ToList();
			hidden_bias_avgs = hidden_bias_avgs.Select(x => x /= bachsize).ToList();
			hidden_output_weight_gradient_avgs = hidden_output_weight_gradient_avgs.Select(x => x /= bachsize).ToList();
			output_bias_avgs = output_bias_avgs.Select(x => x /= bachsize).ToList();
			loss_avg /= bachsize;

			updateWeightsAndBiases(input_hidden_weight_gradient_avgs, hidden_bias_avgs, hidden_output_weight_gradient_avgs, output_bias_avgs);
			if (debug && run_counter % 100 == 0) Console.WriteLine($"loss: {loss_avg}");
		}
		private void updateWeightsAndBiases(List<double> input_hidden_w, List<double> hidden_b, List<double> hidden_output_w, List<double> output_b)
		{
			updateList(_input_hidden_weights, input_hidden_w);
			updateList(_hidden_biases, hidden_b);
			updateList(_hidden_output_weights, hidden_output_w);
			updateList(_output_biases, output_b);
		}
		private void updateList(List<double> to_list, List<double> from_list)
		{
			for (int i = 0; i < to_list.Count; i++)
			{
				to_list[i] = to_list[i] - _lr * from_list[i];
			}
		}
		private double getOutputGradientToWeight(double prediction, double correct_prediction, int true_label, double hidden_activation_output)
		{
			return (prediction - true_label) * hidden_activation_output;
		}
		private double getOutputGradientToWeight(double output_gradient_bias, double hidden_activation_output)
		{
			return output_gradient_bias * hidden_activation_output;
		}
		private double getOutputGradientToBias(double prediction, double correct_prediction, int true_label)
		{
			return prediction - true_label;
		}
		private double getHiddenGradientToBias(double contribution, double reluGradient)
		{
			return contribution * reluGradient;
		}
		private double getHiddenGradientToWeight(double gradientToBias, double input_value)
		{
			return gradientToBias * input_value;
		}
		private double getHiddenLayerContributionToOutput(int hidden_neuron_index, int correct_answer, List<double> activatedOutputLayerValues)
		{
			double sum = 0;
			for(int i = 0; i < _output_biases.Count; i++)
			{
				sum += _hidden_output_weights[i + _output_biases.Count * hidden_neuron_index] * (activatedOutputLayerValues[i] - getKroneckerDelta(i, correct_answer));
			}
			return sum;
		}


		private int getKroneckerDelta(int i, int j)
		{
			return i == j ? 1 : 0;
		}
		private double getReLUGradient(double x)
		{
			return x > 0 ? 1 : 0;
		}	
		private double getLoss(List<double> prediction, List<int> true_label) 
		{
			double loss_sum = 0;
			for (int i = 0; i < prediction.Count; i++) 
			{
				loss_sum -= true_label[i] * Math.Log(prediction[i]);
			}
			return loss_sum;
		}
		private double computeWeightedSum(List<double> weights, List<double> values, double bias, int skip)
		{
			double sum = 0;
			for (int i = 0; i < values.Count; i++) 
			{
				sum += weights[i + values.Count * skip] * values[i];
			}
			sum += bias;
			return sum;
		}
		private List<double> computeHiddenLayerValues(List<double> weights, List<double> values, List<double> biases)
		{
			return computeHiddenLayerValuesRaw(weights, values, biases).Select(hiddenActivationReLU).ToList();
		}
		private List<double> computeHiddenLayerValues(List<double> computed_raw_values)
		{
			return computed_raw_values.Select(hiddenActivationReLU).ToList();
		}
		private List<double> computeHiddenLayerValuesRaw(List<double> weights, List<double> values, List<double> biases)
		{
			List<double> hidden = new List<double>();
			int weight_per_value = weights.Count / values.Count;
			for (int i = 0; i < weight_per_value; i++)
			{
				hidden.Add(computeWeightedSum(weights, values, biases[i], i));
			}
			return hidden;
		}
		private List<double> computeOutputLayerRawValues(List<double> weights, List<double> values, List<double> biases)
		{
			List<double> output = new List<double>();
			int weight_per_value = weights.Count / values.Count;
			for (int i = 0; i < weight_per_value; i++)
			{
				output.Add(computeWeightedSum(weights, values, biases[i], i));
			}
			return output;
		}
		private List<double> computeOutputLayerValues(List<double> computed_raw_values)
		{
			return outputActivationSoftMax(computed_raw_values);
		}
		private List<double> computeOutputLayerValues(List<double> weights, List<double> values, List<double> biases)
		{
			return outputActivationSoftMax(computeOutputLayerRawValues(weights, values, biases));
		}	
		private double hiddenActivationReLU(double value) 
		{
			return Math.Max(0, value);
		}
		private List<double> outputActivationSoftMax(List<double> values) 
		{
			List<double> output = new List<double>();
			double dividend = 0;
            foreach (double val in values)
            {
				dividend += Math.Exp(val);
            }
			foreach (double val in values)
			{
				output.Add(Math.Exp(val) / dividend);
			}
			return output;
        }
		private void Save()
		{
			using (StreamWriter sw = new StreamWriter(Statics.SAVEFILE))
			{
				sw.WriteLine(string.Join(';', _input_hidden_weights));
				sw.WriteLine(string.Join(';', _hidden_output_weights));
				sw.WriteLine(string.Join(';', _hidden_biases));
				sw.WriteLine(string.Join(';', _output_biases));
			}
		}
		private void randomizeWeightsBiases()
		{
			randomizeList(_input_hidden_weights, 1024);
			randomizeList(_hidden_output_weights, 160);
			randomizeList(_hidden_biases, 16);
			randomizeList(_output_biases, 10);
		}
		private void randomizeList(List<double> random_list, int length, double min = -0.5, double max = 0.5) 
		{
			for (int i = 0; i < length; i++) 
			{
				random_list.Add(Random.Shared.NextDouble() * (max - min) + min);
			}
		}
		

	}
}
