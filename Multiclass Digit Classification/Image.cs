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
