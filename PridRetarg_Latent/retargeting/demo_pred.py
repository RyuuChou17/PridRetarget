import os
from datasets.bvh_parser import BVH_file
from datasets.bvh_writer import BVH_writer
from models.IK import fix_foot_contact
from os.path import join as pjoin
import option_parser


# downsampling and remove redundant joints
def copy_ref_file(src, dst, start_frame, end_frame):
    file = BVH_file(src)
    writer = BVH_writer(file.edges, file.names)
    frames = file.to_tensor(quater=True)[..., ::2]
    selected_frames = frames[:,start_frame-1:end_frame]
    writer.write_raw(selected_frames, 'quaternion', dst)


def get_height(file):
    file = BVH_file(file)
    return file.get_height()


def example(src_name, dest_name, bvh_name, test_type, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    parser = option_parser.get_parser()
    args = parser.parse_args()

    input_file = './datasets/Mixamo/{}/{}'.format(src_name, bvh_name)
    ref_file = './datasets/Mixamo/{}/{}'.format(dest_name, bvh_name)
    input_size = args.window_size // 2
    copy_ref_file(input_file, pjoin(output_path, 'input.bvh'), input_size+1, args.window_size)
    copy_ref_file(ref_file, pjoin(output_path, 'gt.bvh'), input_size+1, args.window_size)
    height = get_height(input_file)

    bvh_name = bvh_name.replace(' ', '_')
    input_file = './datasets/Mixamo/{}/{}'.format(src_name, bvh_name)
    ref_file = './datasets/Mixamo/{}/{}'.format(dest_name, bvh_name)

    cmd = 'python eval_single_pair.py --input_bvh={} --target_bvh={} --output_filename={} --test_type={}'.format(
        input_file, ref_file, pjoin(output_path, 'result.bvh'), test_type
    )
    os.system(cmd)

    # fix_foot_contact(pjoin(output_path, 'result.bvh'),
    #                  pjoin(output_path, 'input.bvh'),
    #                  pjoin(output_path, 'result.bvh'),
    #                  height)


if __name__ == '__main__':
    test_list = ['Baseball Pitching',]
                #   'Body Jab Cross', 'Box Turn', 'Idle']

    for test in test_list:
        example('BigVegas', 'Mousey_m', '{}.bvh'.format(test), 'cross', './examples/cross_structure/{}'.format(test).replace(' ', '_'))

    print('Finished!')