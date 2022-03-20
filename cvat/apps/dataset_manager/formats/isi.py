from cProfile import label
from tempfile import TemporaryDirectory
import yaml
import numpy as np

from ..util import make_zip_archive
from .registry import exporter, importer
from ..bindings import TaskData

# Some work around to handle the !!opencv-matrix tag
# see https://stackoverflow.com/questions/36315644/how-do-you-add-a-type-tag-when-using-a-pyyaml-dumper/36317291#answer-36317291
OPENCV_MATRIX_TAG = u"tag:yaml.org,2002:opencv-matrix"

# representer for adding !!opencv_matrix in yaml
class OpenCVMatrixTag(dict):
    pass
def yaml_opencv_matrix_tag_representer(dumper: yaml.SafeDumper, data: OpenCVMatrixTag) -> yaml.nodes.MappingNode:
    return dumper.represent_mapping(tag=OPENCV_MATRIX_TAG, mapping=data.items())

yaml.representer.SafeRepresenter.add_representer(data_type=OpenCVMatrixTag, representer=yaml_opencv_matrix_tag_representer)

# constructor node for !!opencv_matrix
def yaml_opencv_matrix_tag_constuctor(loader: yaml.UnsafeLoader, node: yaml.MappingNode):
    return loader.construct_mapping(node, deep=True)

yaml.add_constructor(tag=OPENCV_MATRIX_TAG, constructor=yaml_opencv_matrix_tag_constuctor)


class MarkerReader:
    def __init__(self, file):
        # Open file
        # Need some work around to make it load by yaml
        # - handle !!opencv-matrix tag has
        # - Ignore comment that may contains ':'

        f = open(file)
        l = f.readline()
        if not (l[0] == "%"):
            f.seek(0)
        self.data = yaml.load(f, Loader=yaml.FullLoader)
        self._num = 0

    @property
    def raw_data(self):
        return np.array(self.data["data"]["data"]).reshape(self.nb_frame, self.data["data"]["cols"])

    @property
    def nb_marker(self):
        return int((self.data["data"]["cols"] - 1) / 2)

    @property
    def nb_frame(self):
        return self.data["data"]["rows"]


@exporter(name='ISI', ext='ZIP', version='1.0')
def _export(file_object, instance_data :TaskData, save_images=False):
    """ Custom exporter that match the SLAM groundtruth format"""
    # Look like exporter extension must be a ZIP file
    # All other are ZIP file, only importer can have another extension
    with TemporaryDirectory() as temp_dir:
        nb_keyframe = 0
        nb_tracks = sum(1 for _ in instance_data.tracks)
        raw_data = []
        for frame_annotation in instance_data.group_by_frame():
            if frame_annotation.labeled_shapes[0].keyframe:
                raw_data.append(frame_annotation.frame)
                nb_keyframe += 1
                for shape in frame_annotation.labeled_shapes:
                    pt = [-1,-1] if shape.occluded else shape.points[0:2]
                    raw_data += pt

        data = {}
        data['rows'] = nb_keyframe
        data['cols'] = 1 + 2 * nb_tracks
        data['dt'] = 'f'
        data['data'] = raw_data
        # wrap in a class to have a custom yaml representer
        data=OpenCVMatrixTag(data)
        data = {'data': data}
        with open(temp_dir+'/groundTruth.yaml', 'w') as f:

            # write the file
            f.write('%YAML:1.0\n')
            f.write('\n')
            yaml.safe_dump(data, f)

        make_zip_archive(temp_dir, file_object)

@importer(name='ISI', ext='yaml', version='1.0')
def _import(src_file, instance_data:TaskData, load_data_callback=None):
    reader = MarkerReader(src_file.name)
    data = reader.raw_data
    for marker_idx in range(reader.nb_marker):
        trackShapes = []
        for frame_idx in range(reader.nb_frame):
            # create and add TrackedShape
            frame = data[frame_idx, 0]
            x = data[frame_idx, 1 + 2*marker_idx]
            y = data[frame_idx, 1 + 2*marker_idx+1]
            occluded = True if (x,y) == (-1,-1) else False
            current_trackShape = TaskData.TrackedShape(
                type='points',
                frame=frame,
                points=[x,y],
                occluded=occluded,
                outside=False,
                keyframe=True,
                attributes=[],
                track_id=marker_idx
            )
            trackShapes.append(current_trackShape)
        # create and add Track
        current_track = TaskData.Track(
            label='Control Point',
            group=0,
            source='imported',
            shapes=trackShapes)
        instance_data.add_track(current_track)