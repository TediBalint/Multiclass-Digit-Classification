using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Multiclass_Digit_Classification
{
	public class DataLoader
	{
		private Dataset dataset;
		public DataLoader(string folderPath, int maxSize, int skip = 0) 
		{
			dataset = new Dataset();
			List<string> dirs = Directory.GetDirectories(folderPath).ToList();
			maxSize /= dirs.Count; skip /= dirs.Count;
			foreach (string dir in dirs) 
			{
				int size = 0; int skipped = 0;

				int label = int.Parse(dir[dir.Length-1].ToString());
                Console.WriteLine(label);
				List<string> files = Directory.GetFiles(dir).ToList();
				foreach (string file in files) 
				{
					skipped++;
					if(skipped < skip) continue;
					size++;
					loadImage(file, label);
				}
				if (size >= maxSize) continue;
			}
		}
		private void loadImage(string filePath, int label)
		{
			dataset.AddImage(new Image(filePath, label));
		}
		public Dataset GetDataset()
		{
			return dataset;
		}
	}
}
