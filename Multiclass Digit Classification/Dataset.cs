using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Multiclass_Digit_Classification
{
	public class Dataset
	{
		private List<Image> _images;
		private int _current_index;
		public Image NextImage { get { return _images[_current_index++]; } }
        public Dataset()
        {
            _current_index = 0;
			_images = new List<Image>();
        }
		public void AddImage(Image image) 
		{
			_images.Add(image);
		}
        public void Shuffle(int amount)
		{
			for (int i = 0; i < amount; i++) shuffleOne();
		}
		public void Shuffle()
		{
			for (int i = 0; i < _images.Count; i++) shuffleOne();
		}
		private void shuffleOne()
		{
			int i = Random.Shared.Next(0, _images.Count);
			int j = Random.Shared.Next(0, _images.Count);
			Image tmp_image = _images[i];
			_images[i] = _images[j];
			_images[j] = tmp_image;
		}
	}
}
