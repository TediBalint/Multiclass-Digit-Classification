using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Multiclass_Digit_Classification
{
	public class NeuralNetwork
	{
		private readonly List<double> _input_hidden_weights;
		private readonly List<double> _output_hidden_weights;
		private readonly List<double> _hidden_biases;
		private readonly List<double> _output_biases;

		private double _lr;

		private DataLoader _loader;


		public NeuralNetwork(double lr, string inputFolder, int datasize)
		{
			_lr = lr;
			_loader = new DataLoader(inputFolder, datasize);
			randomizeWeightsBiases();
		}
		
		private double getOutputGradientToWeight(double prediction, int true_label, double hidden_activation_output)
		{
			return (prediction - true_label) * prediction * (1 - prediction) * hidden_activation_output;
		}
		private double getOutputGradientToBias(double prediction, int true_label)
		{
			return (prediction - true_label) * prediction * (1 - prediction);
		}
		private double getHiddenGradientToWeight(double prediction, int true_label, double input_value, int input_index)
		{
			return getHiddenGradientToActivation(prediction, true_label, input_index) * getReLUGradient(0) * ;
		}


		private double getHiddenGradientToActivation(double prediction, int true_label, int skip)
		{
			double sum = 0;
			for (int i = 0; i < _hidden_biases.Count; i++)
			{
				sum += getOutputGradientToBias(prediction, true_label) * _output_hidden_weights[i + skip * _hidden_biases.Count];
			}
			return sum;
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
		private void randomizeWeightsBiases()
		{
			randomizeList(_input_hidden_weights, 1024);
			randomizeList(_output_hidden_weights, 160);
			randomizeList(_hidden_biases, 16);
			randomizeList(_output_biases, 10);
		}
		private void randomizeList(List<double> random_list, int length, double min = -0.5, double max = 0.5) 
		{
			random_list = new List<double>(random_list);
			for (int i = 0; i < length; i++) 
			{
				random_list.Add(Random.Shared.NextDouble() * (max - min) + min);
			}
		}
		

	}
}
