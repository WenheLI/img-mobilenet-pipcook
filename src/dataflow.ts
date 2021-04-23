import { DataFlowEntry, ScriptContext } from '@pipcook/core';
import { Dataset } from '@pipcook/datacook';

const resizeEntry: DataFlowEntry<Dataset.Types.Sample, Dataset.Types.ImageDatasetMeta> = async (dataset: Dataset.Types.Dataset<Dataset.Types.Sample, any>, options: Record<string, any>, context: ScriptContext)  => {
  const [x='-1', y='-1'] = options['size'];
  const parsedX = parseInt(x);
  const parsedY = parseInt(y);
  if (parsedX == -1 || parsedY == -1) return;

  return Dataset.transformDataset<Dataset.Types.ImageDatasetMeta, Dataset.Types.Sample>({
    next: async (sample) => {
      const resized = sample.data.resize(parsedX, parsedY);
      const { normalize = false } = options;
      if (normalize) return {
        data: context.dataCook.Image.normalize(resized.toTensor()),
        label: sample.label,
      }
      return {
        data: resized.toTensor(),
        label: sample.label
      }
    },
    metadata: async (meta) => {
      return {
        ...meta,
        dimension: {
          x: parsedX,
          y: parsedY,
          z: meta.dimension.z
        }
      }
    }
  }, dataset)
}

/**
 * this is the data process plugin to process pasvoc format data. It supports resize the image and normalize the image
 * @param resize =[256, 256][optional] resize all images to same size
 * @param normalize =false[optional] if normalize all images to have values between [0, 1]
 */
export default resizeEntry;



