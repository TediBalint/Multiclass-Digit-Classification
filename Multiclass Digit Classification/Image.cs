using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Drawing;

namespace Multiclass_Digit_Classification
{
	public class Image
	{
		private List<double> _image_data;
		private int _label;
		public int Label { get { return _label; } }
		public List<double> ImageData {  get { return _image_data; } }

		public Image(string filePath, int label)
		{
			_label = label;
			_image_data = new List<double>();
			setImageData(filePath);
		}
		public void Show()
		{
			Console.Clear();
			Console.SetCursorPosition(0, 0);

			int n = (int)Math.Sqrt(ImageData.Count);

			for (int i = 0; i < n; i++)
			{
				for (int j = 0; j < n; j++)
				{
					if (ImageData[i * n + j] > 0.2) Console.BackgroundColor = ConsoleColor.White;
					else Console.BackgroundColor = ConsoleColor.Black;
					Console.Write(" ");
				}
				Console.WriteLine();
			}
			Console.SetCursorPosition(0, 10);
		}
		private void setImageData(string filePath)
		{

			using (Bitmap bitmap = new Bitmap(filePath))
			{
				using (Bitmap resized = new Bitmap(bitmap, new Size(8, 8)))
				{
					int width = resized.Width;
					int height = resized.Height;
					for (int y = 0; y < height; y++)
					{
						for (int x = 0; x < width; x++)
						{
							_image_data.Add(resized.GetPixel(x, y).GetBrightness());
						}
					}
				}
			}
		}
	}
}
