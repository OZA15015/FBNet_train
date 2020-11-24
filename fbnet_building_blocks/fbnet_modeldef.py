# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# please, end the file with '}' and nothing else. this file updated automatically

MODEL_ARCH = {
    # FBNet-A
    "fbnet_a": {
        "block_op_type": [
            ["skip"],                                               # stage 1
            ["ir_k3_e3"], ["ir_k3_e1"], ["skip"],     ["skip"],     # stage 2
            ["ir_k5_e6"], ["ir_k3_e3"], ["ir_k5_e1"], ["ir_k3_e3"], # stage 3
            ["ir_k5_e6"], ["ir_k5_e3"], ["ir_k5_s2"], ["ir_k5_e6"], # stage 4
            ["ir_k3_e6"], ["ir_k5_s2"], ["ir_k5_e1"], ["ir_k3_s2"], # stage 5
            ["ir_k5_e6"], ["ir_k5_e6"], ["ir_k5_e3"], ["ir_k5_e6"], # stage 6
            ["ir_k5_e6"],                                           # stage 7
        ],
        "block_cfg": {
            # [channel, stride] or [channel, stride, kernel]
            "first": [16, 1],
            "stages": [
                # [[expantion_ratio, channel, number_of_layers, stride_at_first_block_of_stage]]
                [[3, 16, 1, 1]],                                                        # stage 1
                [[3, 24, 1, 2]],  [[6, 24, 1, 1]],      [[1, 24, 1, 1]],  [[6, 24, 1, 1]],  # stage 2
                [[6, 32, 1, 2]],  [[6, 32, 1, 1]],      [[6, 32, 1, 1]],  [[1, 32, 1, 1]],  # stage 3
                [[6, 64, 1, 2]],  [[1, 64, 1, 1]],      [[1, 64, 1, 1]],  [[1, 64, 1, 1]],  # stage 4
                [[6, 96, 1, 1]], [[1, 96, 1, 1]],     [[1, 96, 1, 1]], [[1, 96, 1, 1]], # stage 5
                [[1, 160, 1, 2]], [[1, 160, 1, 1]],     [[1, 160, 1, 1]], [[1, 160, 1, 1]], # stage 6
                [[1, 320, 1, 1]],   
                
            ],
            "backbone": [num for num in range(23)],
        },
    },
    # FBNet-B
    "fbnet_b": {
        "block_op_type": [
            ["ir_k3_e1"],                                           # stage 1
            ["ir_k3_e6"], ["ir_k5_e1"], ["ir_k3_e1"], ["ir_k3_e1"], # stage 2
            ["ir_k5_e6"], ["ir_k5_e3"], ["ir_k3_e6"], ["ir_k5_e6"], # stage 3
            ["ir_k5_e6"], ["ir_k5_e1"], ["skip"],     ["ir_k5_e3"], # stage 4
            ["ir_k5_e6"], ["ir_k3_e1"], ["ir_k5_e1"], ["ir_k5_e3"], # stage 5
            ["ir_k5_e6"], ["ir_k5_e1"], ["ir_k5_e6"], ["ir_k5_e6"], # stage 6
            ["ir_k3_e6"],                                           # stage 7
        ],
        "block_cfg": {
            # [channel, stride] or [channel, stride, kernel]
            "first": [16, 2],
            "stages": [
                # [[expantion_ratio, channel, number_of_layers, stride_at_first_block_of_stage]]
                [[1, 16, 1, 1]],                                                        # stage 1
                [[3, 24, 1, 2]],  [[1, 24, 1, 1]],  [[1, 24, 1, 1]],  [[1, 24, 1, 1]],  # stage 2
                [[6, 32, 1, 2]],  [[3, 32, 1, 1]],  [[6, 32, 1, 1]],  [[6, 32, 1, 1]],  # stage 3
                [[6, 64, 1, 2]],  [[1, 64, 1, 1]],  [[1, 64, 1, 1]],  [[3, 64, 1, 1]],  # stage 4
                [[6, 112, 1, 1]], [[1, 112, 1, 1]], [[3, 112, 1, 1]], [[3, 112, 1, 1]], # stage 5
                [[6, 184, 1, 2]], [[1, 184, 1, 1]], [[6, 184, 1, 1]], [[6, 184, 1, 1]], # stage 6
                [[6, 352, 1, 1]],                                                       # stage 7
            ],
            "backbone": [num for num in range(23)],
        },
    },
    # FBNet-C
    "fbnet_c": {
        "block_op_type": [
            ["ir_k3_e1"],                                           # stage 1
            ["ir_k3_e6"], ["skip"],     ["ir_k3_e1"], ["ir_k3_e1"], # stage 2
            ["ir_k5_e6"], ["ir_k5_e3"], ["ir_k5_e6"], ["ir_k3_e6"], # stage 3
            ["ir_k5_e6"], ["ir_k5_e3"], ["ir_k5_e6"], ["ir_k5_e6"], # stage 4
            ["ir_k5_e6"], ["ir_k5_e6"], ["ir_k5_e6"], ["ir_k5_e3"], # stage 5
            ["ir_k5_e6"], ["ir_k5_e6"], ["ir_k5_e6"], ["ir_k5_e6"], # stage 6
            ["ir_k3_e6"],                                           # stage 7
        ],
        "block_cfg": {
            # [channel, stride] or [channel, stride, kernel]
            "first": [16, 1],
            "stages": [
                # [[expantion_ratio, channel, number_of_layers, stride_at_first_block_of_stage]]
                [[1, 16, 1, 1]],                                                        # stage 1
                [[3, 24, 1, 2]],  [[1, 24, 1, 1]],  [[1, 24, 1, 1]],  [[1, 24, 1, 1]],  # stage 2
                [[6, 32, 1, 2]],  [[3, 32, 1, 1]],  [[6, 32, 1, 1]],  [[6, 32, 1, 1]],  # stage 3
                [[6, 64, 1, 2]],  [[3, 64, 1, 1]],  [[6, 64, 1, 1]],  [[6, 64, 1, 1]],  # stage 4
                [[6, 112, 1, 1]], [[6, 112, 1, 1]], [[6, 112, 1, 1]], [[3, 112, 1, 1]], # stage 5
                [[6, 184, 1, 2]], [[6, 184, 1, 1]], [[6, 184, 1, 1]], [[6, 184, 1, 1]], # stage 6
                [[6, 352, 1, 1]],                                                       # stage 7
            ],
            "backbone": [num for num in range(23)],
        },
    },
    # FBNet-96-0.35-1 - for input size 96 and channel scaling 0.35
    "fbnet_96_035_1": {
        "block_op_type": [
            ["ir_k3_e1"],                                   # stage 1
            ["ir_k3_e6"], ["ir_k3_e6"], ["skip"], ["skip"], # stage 2
            ["ir_k5_e6"], ["ir_k5_e6"], ["skip"], ["skip"], # stage 3
            ["ir_k5_e6"], ["skip"],     ["skip"], ["skip"], # stage 4
            ["ir_k3_e6"], ["skip"],     ["skip"], ["skip"], # stage 5
            ["ir_k5_e6"], ["ir_k5_e6"], ["skip"], ["skip"], # stage 6
            ["ir_k5_e6"],                                   # stage 7
        ],
        "block_cfg": {
            # [channel, stride] or [channel, stride, kernel]
            "first": [16, 2],
            "stages": [
                # [[expantion_ratio, channel, number_of_layers, stride_at_first_block_of_stage]]
                [[1, 16, 1, 1]],                                                        # stage 1
                [[6, 24, 1, 2]],  [[6, 24, 1, 1]],  [[1, 24, 1, 1]],  [[1, 24, 1, 1]],  # stage 2
                [[6, 32, 1, 2]],  [[6, 32, 1, 1]],  [[1, 32, 1, 1]],  [[1, 32, 1, 1]],  # stage 3
                [[6, 64, 1, 2]],  [[1, 64, 1, 1]],  [[1, 64, 1, 1]],  [[1, 64, 1, 1]],  # stage 4
                [[6, 112, 1, 1]], [[1, 112, 1, 1]], [[1, 112, 1, 1]], [[1, 112, 1, 1]], # stage 5
                [[6, 184, 1, 2]], [[6, 184, 1, 1]], [[1, 184, 1, 1]], [[1, 184, 1, 1]], # stage 6
                [[6, 352, 1, 1]],                                                       # stage 7
            ],
            "backbone": [num for num in range(23)],
        },
    },
    # FBNet-Samsung-S8
    "fbnet_samsung_s8": {
        "block_op_type": [
            ["ir_k3_e1"],                                           # stage 1
            ["ir_k3_e3"], ["ir_k3_e1"], ["skip"],     ["skip"],     # stage 2
            ["ir_k5_e6"], ["ir_k3_e3"], ["ir_k3_e3"], ["ir_k3_e6"], # stage 3
            ["ir_k5_e6"], ["ir_k5_e3"], ["ir_k5_e3"], ["ir_k5_e3"], # stage 4
            ["ir_k3_e6"], ["ir_k5_e3"], ["ir_k5_e3"], ["ir_k5_e3"], # stage 5
            ["ir_k5_e6"], ["ir_k5_e6"], ["ir_k5_e3"], ["ir_k5_e6"], # stage 6
            ["ir_k5_e6"],                                           # stage 7
        ],
        "block_cfg": {
            # [channel, stride] or [channel, stride, kernel]
            "first": [16, 2],
            "stages": [
                # [[expantion_ratio, channel, number_of_layers, stride_at_first_block_of_stage]]
                [[1, 16, 1, 1]],                                                        # stage 1
                [[3, 24, 1, 2]],  [[1, 24, 1, 1]],  [[1, 24, 1, 1]],  [[1, 24, 1, 1]],  # stage 2
                [[6, 32, 1, 2]],  [[3, 32, 1, 1]],  [[3, 32, 1, 1]],  [[6, 32, 1, 1]],  # stage 3
                [[6, 64, 1, 2]],  [[3, 64, 1, 1]],  [[3, 64, 1, 1]],  [[3, 64, 1, 1]],  # stage 4
                [[6, 112, 1, 1]], [[3, 112, 1, 1]], [[3, 112, 1, 1]], [[3, 112, 1, 1]], # stage 5
                [[6, 184, 1, 2]], [[6, 184, 1, 1]], [[3, 184, 1, 1]], [[6, 184, 1, 1]], # stage 6
                [[6, 352, 1, 1]],                                                       # stage 7
            ],
            "backbone": [num for num in range(23)],
        },
    },
    # FBNet-iPhoneX
    "fbnet_iphonex": {
        "block_op_type": [
            ["skip"],                                               # stage 1
            ["ir_k3_e6"], ["ir_k3_e1"], ["ir_k3_e1"], ["ir_k3_e1"], # stage 2
            ["ir_k3_e6"], ["ir_k3_e6"], ["ir_k3_e6"], ["ir_k3_e6"], # stage 3
            ["ir_k3_e6"], ["ir_k3_e6"], ["ir_k3_e3"], ["ir_k3_e6"], # stage 4
            ["ir_k3_e6"], ["ir_k3_e3"], ["ir_k3_e3"], ["ir_k3_e3"], # stage 5
            ["ir_k5_e6"], ["ir_k5_e6"], ["ir_k5_e6"], ["ir_k5_e3"], # stage 6
            ["ir_k5_e6"],                                           # stage 7
        ],
        "block_cfg": {
            # [channel, stride] or [channel, stride, kernel]
            "first": [16, 2],
            "stages": [
                # [[expantion_ratio, channel, number_of_layers, stride_at_first_block_of_stage]]
                [[1, 16, 1, 1]],                                                        # stage 1
                [[3, 24, 1, 2]],  [[1, 24, 1, 1]],  [[1, 24, 1, 1]],  [[1, 24, 1, 1]],  # stage 2
                [[6, 32, 1, 2]],  [[6, 32, 1, 1]],  [[6, 32, 1, 1]],  [[6, 32, 1, 1]],  # stage 3
                [[6, 64, 1, 2]],  [[6, 64, 1, 1]],  [[3, 64, 1, 1]],  [[6, 64, 1, 1]],  # stage 4
                [[6, 112, 1, 1]], [[3, 112, 1, 1]], [[3, 112, 1, 1]], [[3, 112, 1, 1]], # stage 5
                [[6, 184, 1, 2]], [[6, 184, 1, 1]], [[6, 184, 1, 1]], [[3, 184, 1, 1]], # stage 6
                [[6, 352, 1, 1]],                                                       # stage 7
            ],
            "backbone": [num for num in range(23)],
        },
    },
    # Searched Architecture
    "fbnet_cpu_sample1": {
        "block_op_type": [
            ["ir_k5_e6"],
            ["ir_k3_e6"], ["ir_k5_e6"], ["ir_k5_e6"], ["ir_k5_e6"],
            ["ir_k3_e6"], ["ir_k5_e6"], ["ir_k5_e3"], ["skip"],
            ["ir_k5_e6"], ["ir_k3_e6"], ["skip"], ["skip"],
            ["ir_k5_e6"], ["ir_k5_e6"], ["skip"], ["skip"],
            ["ir_k5_e6"], ["skip"], ["skip"], ["skip"],
            ["skip"],
        ],
        "block_cfg": {
            "first": [16, 2],
            "stages": [
                [[6, 16, 1, 1]],                                                        # stage 1
                [[6, 24, 1, 2]],  [[6, 24, 1, 1]],  [[6, 24, 1, 1]],  [[6, 24, 1, 1]],  # stage 2
                [[6, 32, 1, 2]],  [[6, 32, 1, 1]],  [[3, 32, 1, 1]],  [[1, 32, 1, 1]],  # stage 3
                [[6, 64, 1, 2]],  [[6, 64, 1, 1]],  [[1, 64, 1, 1]],  [[1, 64, 1, 1]],  # stage 4
                [[6, 112, 1, 1]], [[6, 112, 1, 1]], [[1, 112, 1, 1]], [[1, 112, 1, 1]], # stage 5
                [[6, 184, 1, 2]], [[1, 184, 1, 1]], [[1, 184, 1, 1]], [[1, 184, 1, 1]], # stage 6
                [[1, 352, 1, 1]],                                                       # stage 7
            ],
            "backbone": [num for num in range(23)],
        },
    },
    "fbnet_cpu_sample2": {
            "block_op_type": [
            ["ir_k5_e6"], 
            ["ir_k3_e6"], ["ir_k3_e6"], ["ir_k5_e6"], ["ir_k5_e6"], 
            ["ir_k5_e6"], ["ir_k5_e6"], ["skip"], ["skip"], 
            ["ir_k3_e6"], ["ir_k5_e6"], ["skip"], ["skip"], 
            ["ir_k5_e6"], ["skip"], ["skip"], ["skip"], 
            ["ir_k5_e6"], ["skip"], ["skip"], ["skip"], 
            ["skip"], 
            ],
            "block_cfg": {
                "first": [16, 2],
                "stages": [
                    [[6, 16, 1, 1]],                                                            # stage 1
                    [[6, 24, 1, 2]],  [[6, 24, 1, 1]],      [[6, 24, 1, 1]],  [[6, 24, 1, 1]],  # stage 2
                    [[6, 32, 1, 2]],  [[6, 32, 1, 1]],      [[1, 32, 1, 1]],  [[1, 32, 1, 1]],  # stage 3
                    [[6, 64, 1, 2]],  [[6, 64, 1, 1]],      [[1, 64, 1, 1]],  [[1, 64, 1, 1]],  # stage 4
                    [[6, 112, 1, 1]], [[1, 112, 1, 1]],     [[1, 112, 1, 1]], [[1, 112, 1, 1]], # stage 5
                    [[6, 184, 1, 2]], [[1, 184, 1, 1]],     [[1, 184, 1, 1]], [[1, 184, 1, 1]], # stage 6
                    [[1, 352, 1, 1]],                                                           # stage 7
                ],
                "backbone": [num for num in range(23)],
            },
        },
    "my_unique_name_for_architecture": {
            "block_op_type": [
            ["ir_k5_e6"], 
            ["ir_k5_e6"], ["ir_k5_e6"], ["ir_k5_e6"], ["ir_k5_e6"], 
            ["ir_k5_e6"], ["ir_k5_e6"], ["ir_k5_e6"], ["ir_k5_e6"], 
            ["ir_k5_e6"], ["ir_k3_e6"], ["ir_k5_e6"], ["skip"], 
            ["ir_k3_e3"], ["ir_k5_e6"], ["ir_k5_e6"], ["skip"], 
            ["ir_k3_e6"], ["ir_k5_e3"], ["skip"], ["skip"], 
            ["skip"], 
            ],
            "block_cfg": {
                "first": [16, 2],
                "stages": [
                    [[6, 16, 1, 1]],                                                        # stage 1
                    [[6, 24, 1, 2]],  [[6, 24, 1, 1]],      [[6, 24, 1, 1]],  [[6, 24, 1, 1]],  # stage 2
                    [[6, 32, 1, 2]],  [[6, 32, 1, 1]],      [[6, 32, 1, 1]],  [[6, 32, 1, 1]],  # stage 3
                    [[6, 64, 1, 2]],  [[6, 64, 1, 1]],      [[6, 64, 1, 1]],  [[1, 64, 1, 1]],  # stage 4
                    [[3, 112, 1, 1]], [[6, 112, 1, 1]],     [[6, 112, 1, 1]], [[1, 112, 1, 1]], # stage 5
                    [[6, 184, 1, 2]], [[3, 184, 1, 1]],     [[1, 184, 1, 1]], [[1, 184, 1, 1]], # stage 6
                    [[1, 352, 1, 1]],                                                       # stage 7
                ],
                "backbone": [num for num in range(23)],
            },
        },
    "first_test": {
            "block_op_type": [
            ["ir_k3_e6"], 
            ["ir_k5_e6"], ["ir_k5_e6"], ["ir_k3_e6"], ["ir_k5_e6"], 
            ["ir_k5_e6"], ["ir_k5_e6"], ["ir_k5_e6"], ["ir_k5_e6"], 
            ["ir_k5_e6"], ["skip"], ["skip"], ["skip"], 
            ["ir_k3_e6"], ["skip"], ["skip"], ["skip"], 
            ["ir_k5_e6"], ["ir_k5_e3"], ["ir_k3_e1"], ["ir_k5_e1"], 
            ["skip"], 
            ],
            "block_cfg": {
                "first": [16, 1],
                "stages": [
                    [[6, 16, 1, 1]],                                                        # stage 1
                    [[6, 24, 1, 2]],  [[6, 24, 1, 1]],      [[6, 24, 1, 1]],  [[6, 24, 1, 1]],  # stage 2
                    [[6, 32, 1, 2]],  [[6, 32, 1, 1]],      [[6, 32, 1, 1]],  [[6, 32, 1, 1]],  # stage 3
                    [[6, 64, 1, 2]],  [[1, 64, 1, 1]],      [[1, 64, 1, 1]],  [[1, 64, 1, 1]],  # stage 4
                    [[6, 112, 1, 1]], [[1, 112, 1, 1]],     [[1, 112, 1, 1]], [[1, 112, 1, 1]], # stage 5
                    [[6, 184, 1, 2]], [[3, 184, 1, 1]],     [[1, 184, 1, 1]], [[1, 184, 1, 1]], # stage 6
                    [[1, 352, 1, 1]],                                                       # stage 7
                ],
                "backbone": [num for num in range(23)],
            },
        },
    "second_test": {
            "block_op_type": [
            ["ir_k3_e6"], 
            ["ir_k3_e6"], ["ir_k5_e6"], ["ir_k5_e6"], ["ir_k5_e6"], 
            ["ir_k5_e6"], ["ir_k5_e6"], ["ir_k5_e6"], ["ir_k5_e6"], 
            ["ir_k5_e6"], ["skip"], ["skip"], ["skip"], 
            ["ir_k3_e6"], ["skip"], ["skip"], ["skip"], 
            ["ir_k3_e6"], ["skip"], ["ir_k3_s2"], ["ir_k5_s2"], 
            ["skip"], 
            ],
            "block_cfg": {
                "first": [16, 1],
                "stages": [
                    [[6, 16, 1, 1]],                                                        # stage 1
                    [[6, 24, 1, 2]],  [[6, 24, 1, 1]],      [[6, 24, 1, 1]],  [[6, 24, 1, 1]],  # stage 2
                    [[6, 32, 1, 2]],  [[6, 32, 1, 1]],      [[6, 32, 1, 1]],  [[6, 32, 1, 1]],  # stage 3
                    [[6, 64, 1, 2]],  [[1, 64, 1, 1]],      [[1, 64, 1, 1]],  [[1, 64, 1, 1]],  # stage 4
                    [[6, 112, 1, 1]], [[1, 112, 1, 1]],     [[1, 112, 1, 1]], [[1, 112, 1, 1]], # stage 5
                    [[6, 184, 1, 2]], [[1, 184, 1, 1]],     [[1, 184, 1, 1]], [[1, 184, 1, 1]], # stage 6
                    [[1, 352, 1, 1]],                                                       # stage 7
                ],
                "backbone": [num for num in range(23)],
            },
        },

    "third_test": {
            "block_op_type": [
            ["ir_k3_e6"], 
            ["ir_k5_e6"], ["ir_k3_e6"], ["ir_k5_e6"], ["ir_k5_e6"], 
            ["ir_k5_e6"], ["ir_k5_e6"], ["ir_k5_e6"], ["ir_k5_e6"], 
            ["ir_k3_e6"], ["skip"], ["skip"], ["skip"], 
            ["skip"], ["skip"], ["skip"], ["skip"], 
            ["ir_k5_e3"], ["skip"], ["ir_k3_e1"], ["ir_k5_s2"], 
            ["skip"], 
            ],
            "block_cfg": {
                "first": [16, 1],
                "stages": [
                    [[6, 16, 1, 1]],                                                        # stage 1
                    [[6, 24, 1, 2]],  [[6, 24, 1, 1]],      [[6, 24, 1, 1]],  [[6, 24, 1, 1]],  # stage 2
                    [[6, 32, 1, 2]],  [[6, 32, 1, 1]],      [[6, 32, 1, 1]],  [[6, 32, 1, 1]],  # stage 3
                    [[6, 64, 1, 2]],  [[1, 64, 1, 1]],      [[1, 64, 1, 1]],  [[1, 64, 1, 1]],  # stage 4
                    [[1, 112, 1, 1]], [[1, 112, 1, 1]],     [[1, 112, 1, 1]], [[1, 112, 1, 1]], # stage 5
                    [[3, 184, 1, 2]], [[1, 184, 1, 1]],     [[1, 184, 1, 1]], [[1, 184, 1, 1]], # stage 6
                    [[1, 352, 1, 1]],                                                       # stage 7
                ],
                "backbone": [num for num in range(23)],
            },
        },

    "third_test2": {
            "block_op_type": [
            ["ir_k5_e6"], 
            ["ir_k5_e6"], ["ir_k3_e6"], ["ir_k3_e6"], ["ir_k5_e6"], 
            ["ir_k5_e6"], ["ir_k5_e6"], ["ir_k3_e6"], ["ir_k5_e6"], 
            ["ir_k5_e6"], ["ir_k3_e6"], ["ir_k5_e6"], ["skip"], 
            ["ir_k5_e6"], ["skip"], ["ir_k5_e6"], ["skip"], 
            ["ir_k5_e6"], ["skip"], ["ir_k3_e3"], ["ir_k5_s2"], 
            ["ir_k3_e3"], 
            ],
            "block_cfg": {
                "first": [16, 1],
                "stages": [
                    [[6, 16, 1, 1]],                                                        # stage 1
                    [[6, 24, 1, 2]],  [[6, 24, 1, 1]],      [[6, 24, 1, 1]],  [[6, 24, 1, 1]],  # stage 2
                    [[6, 32, 1, 2]],  [[6, 32, 1, 1]],      [[6, 32, 1, 1]],  [[6, 32, 1, 1]],  # stage 3
                    [[6, 64, 1, 2]],  [[6, 64, 1, 1]],      [[6, 64, 1, 1]],  [[1, 64, 1, 1]],  # stage 4
                    [[6, 112, 1, 1]], [[1, 112, 1, 1]],     [[6, 112, 1, 1]], [[1, 112, 1, 1]], # stage 5
                    [[6, 184, 1, 2]], [[1, 184, 1, 1]],     [[3, 184, 1, 1]], [[1, 184, 1, 1]], # stage 6
                    [[3, 352, 1, 1]],                                                       # stage 7
                ],
                "backbone": [num for num in range(23)],
            },
        }, 
  
   
    "Normal1110": {
            "block_op_type": [
            ["ir_k3_e6"], 
            ["ir_k3_e6"], ["ir_k5_e6"], ["ir_k5_e6"], ["ir_k5_e6"], 
            ["ir_k5_e6"], ["ir_k5_e6"], ["ir_k5_e6"], ["ir_k3_e6"], 
            ["ir_k5_e6"], ["ir_k5_e6"], ["skip"], ["skip"], 
            ["ir_k5_e6"], ["ir_k5_e6"], ["ir_k3_s2"], ["ir_k3_e6"], 
            ["ir_k5_e6"], ["ir_k5_s2"], ["ir_k5_e1"], ["ir_k5_e6"], 
            ["skip"], 
            ],
            "block_cfg": {
                "first": [16, 1],
                "stages": [
                    [[6, 16, 1, 1]],                                                        # stage 1
                    [[6, 24, 1, 2]],  [[6, 24, 1, 1]],      [[6, 24, 1, 1]],  [[6, 24, 1, 1]],  # stage 2
                    [[6, 32, 1, 2]],  [[6, 32, 1, 1]],      [[6, 32, 1, 1]],  [[6, 32, 1, 1]],  # stage 3
                    [[6, 64, 1, 2]],  [[6, 64, 1, 1]],      [[1, 64, 1, 1]],  [[1, 64, 1, 1]],  # stage 4
                    [[6, 112, 1, 1]], [[6, 112, 1, 1]],     [[1, 112, 1, 1]], [[6, 112, 1, 1]], # stage 5
                    [[6, 184, 1, 2]], [[1, 184, 1, 1]],     [[1, 184, 1, 1]], [[6, 184, 1, 1]], # stage 6
                    [[1, 352, 1, 1]],                                                       # stage 7
                ],
                "backbone": [num for num in range(23)],
            },
        },
    
    "1120test1": {
            "block_op_type": [
            ["ir_k3_e6"], 
            ["ir_k3_e6"], ["ir_k5_e6"], ["ir_k5_e6"], ["ir_k5_e6"], 
            ["ir_k3_e6"], ["ir_k5_e6"], ["ir_k3_e6"], ["ir_k3_e6"], 
            ["ir_k5_e6"], ["ir_k3_e6"], ["skip"], ["skip"], 
            ["ir_k5_e6"], ["skip"], ["ir_k5_e1"], ["skip"], 
            ["skip"], ["skip"], ["skip"], ["ir_k3_e3"], 
            ["ir_k5_e1"], 
            ],
            "block_cfg": {
                "first": [16, 1],
                "stages": [
                    [[6, 16, 1, 1]],                                                        # stage 1
                    [[6, 24, 1, 2]],  [[6, 24, 1, 1]],      [[6, 24, 1, 1]],  [[6, 24, 1, 1]],  # stage 2
                    [[6, 32, 1, 2]],  [[6, 32, 1, 1]],      [[6, 32, 1, 1]],  [[6, 32, 1, 1]],  # stage 3
                    [[6, 64, 1, 2]],  [[6, 64, 1, 1]],      [[1, 64, 1, 1]],  [[1, 64, 1, 1]],  # stage 4
                    [[6, 96, 1, 1]], [[1, 96, 1, 1]],     [[1, 96, 1, 1]], [[1, 96, 1, 1]], # stage 5
                    [[1, 160, 1, 2]], [[1, 160, 1, 1]],     [[1, 160, 1, 1]], [[3, 160, 1, 1]], # stage 6
                    [[1, 320, 1, 1]],                                                       # stage 7
                ],
                "backbone": [num for num in range(23)],
            },
        },

    "1120test3": {
            "block_op_type": [
            ["skip"], 
            ["ir_k5_e6"], ["ir_k5_e3"], ["skip"], ["skip"], 
            ["ir_k3_e6"], ["skip"], ["skip"], ["skip"], 
            ["ir_k3_e6"], ["skip"], ["skip"], ["skip"], 
            ["ir_k3_e6"], ["skip"], ["skip"], ["skip"], 
            ["ir_k3_e1"], ["skip"], ["skip"], ["skip"], 
            ["skip"], 
            ],
            "block_cfg": {
                "first": [16, 1],
                "stages": [
                    [[1, 16, 1, 1]],                                                        # stage 1
                    [[6, 24, 1, 2]],  [[3, 24, 1, 1]],      [[1, 24, 1, 1]],  [[1, 24, 1, 1]],  # stage 2
                    [[6, 32, 1, 2]],  [[1, 32, 1, 1]],      [[1, 32, 1, 1]],  [[1, 32, 1, 1]],  # stage 3
                    [[6, 64, 1, 2]],  [[1, 64, 1, 1]],      [[1, 64, 1, 1]],  [[1, 64, 1, 1]],  # stage 4
                    [[6, 96, 1, 1]], [[1, 96, 1, 1]],     [[1, 96, 1, 1]], [[1, 96, 1, 1]], # stage 5
                    [[1, 160, 1, 2]], [[1, 160, 1, 1]],     [[1, 160, 1, 1]], [[1, 160, 1, 1]], # stage 6
                    [[1, 320, 1, 1]],                                                       # stage 7
                ],
                "backbone": [num for num in range(23)],
            },
        },


    "1120test2": {
            "block_op_type": [
            ["ir_k5_e3"], 
            ["ir_k3_e3"], ["ir_k3_e6"], ["skip"], ["ir_k5_e6"], 
            ["ir_k5_e6"], ["ir_k5_e6"], ["ir_k5_e6"], ["skip"], 
            ["ir_k3_e6"], ["skip"], ["skip"], ["skip"], 
            ["ir_k3_e6"], ["skip"], ["skip"], ["skip"], 
            ["ir_k3_s2"], ["skip"], ["skip"], ["skip"], 
            ["ir_k5_e1"], 
            ],
            "block_cfg": {
                "first": [16, 1],
                "stages": [
                    [[3, 16, 1, 1]],                                                        # stage 1
                    [[3, 24, 1, 2]],  [[6, 24, 1, 1]],      [[1, 24, 1, 1]],  [[6, 24, 1, 1]],  # stage 2
                    [[6, 32, 1, 2]],  [[6, 32, 1, 1]],      [[6, 32, 1, 1]],  [[1, 32, 1, 1]],  # stage 3
                    [[6, 64, 1, 2]],  [[1, 64, 1, 1]],      [[1, 64, 1, 1]],  [[1, 64, 1, 1]],  # stage 4
                    [[6, 96, 1, 1]], [[1, 96, 1, 1]],     [[1, 96, 1, 1]], [[1, 96, 1, 1]], # stage 5
                    [[1, 160, 1, 2]], [[1, 160, 1, 1]],     [[1, 160, 1, 1]], [[1, 160, 1, 1]], # stage 6
                    [[1, 320, 1, 1]],                                                       # stage 7
                ],
                "backbone": [num for num in range(23)],
            },
        },

    "1120test1-new": {
            "block_op_type": [
            ["ir_k3_e6"], 
            ["ir_k5_e6"], ["ir_k5_e6"], ["ir_k5_e6"], ["ir_k5_e6"], 
            ["ir_k5_e6"], ["ir_k5_e6"], ["ir_k5_e6"], ["ir_k5_e6"], 
            ["ir_k5_e6"], ["ir_k5_e6"], ["ir_k5_e6"], ["ir_k5_e6"], 
            ["ir_k5_e6"], ["ir_k5_e6"], ["skip"], ["skip"], 
            ["ir_k5_e6"], ["ir_k3_e6"], ["ir_k3_s2"], ["ir_k5_s2"], 
            ["skip"], 
            ],
            "block_cfg": {
                "first": [16, 1],
                "stages": [
                    [[6, 16, 1, 1]],                                                        # stage 1
                    [[6, 24, 1, 2]],  [[6, 24, 1, 1]],      [[6, 24, 1, 1]],  [[6, 24, 1, 1]],  # stage 2
                    [[6, 32, 1, 2]],  [[6, 32, 1, 1]],      [[6, 32, 1, 1]],  [[6, 32, 1, 1]],  # stage 3
                    [[6, 64, 1, 2]],  [[6, 64, 1, 1]],      [[6, 64, 1, 1]],  [[6, 64, 1, 1]],  # stage 4
                    [[6, 96, 1, 1]], [[6, 96, 1, 1]],     [[1, 96, 1, 1]], [[1, 96, 1, 1]], # stage 5
                    [[6, 160, 1, 2]], [[6, 160, 1, 1]],     [[1, 160, 1, 1]], [[1, 160, 1, 1]], # stage 6
                    [[1, 320, 1, 1]],                                                       # stage 7
                ],
                "backbone": [num for num in range(23)],
            },
        },

    "'fbnet_b'": {
            "block_op_type": [
            ["ir_k3_e3"], 
            ["ir_k3_e6"], ["ir_k3_e3"], ["ir_k5_e6"], ["ir_k5_e6"], 
            ["ir_k5_e6"], ["ir_k5_e6"], ["ir_k5_e6"], ["ir_k5_e6"], 
            ["ir_k5_e6"], ["ir_k5_e6"], ["skip"], ["skip"], 
            ["ir_k5_e6"], ["skip"], ["skip"], ["skip"], 
            ["ir_k5_e3"], ["skip"], ["skip"], ["skip"], 
            ["ir_k3_e3"], 
            ],
            "block_cfg": {
                "first": [16, 1],
                "stages": [
                    [[3, 16, 1, 1]],                                                        # stage 1
                    [[6, 24, 1, 2]],  [[3, 24, 1, 1]],      [[6, 24, 1, 1]],  [[6, 24, 1, 1]],  # stage 2
                    [[6, 32, 1, 2]],  [[6, 32, 1, 1]],      [[6, 32, 1, 1]],  [[6, 32, 1, 1]],  # stage 3
                    [[6, 64, 1, 2]],  [[6, 64, 1, 1]],      [[1, 64, 1, 1]],  [[1, 64, 1, 1]],  # stage 4
                    [[6, 96, 1, 1]], [[1, 96, 1, 1]],     [[1, 96, 1, 1]], [[1, 96, 1, 1]], # stage 5
                    [[3, 160, 1, 2]], [[1, 160, 1, 1]],     [[1, 160, 1, 1]], [[1, 160, 1, 1]], # stage 6
                    [[3, 320, 1, 1]],                                                       # stage 7
                ],
                "backbone": [num for num in range(23)],
            },
        },

    "1123test3_25epoch": {
            "block_op_type": [
            ["skip"], 
            ["ir_k3_e6"], ["skip"], ["skip"], ["ir_k3_s2"], 
            ["ir_k5_e6"], ["skip"], ["skip"], ["skip"], 
            ["ir_k3_e6"], ["skip"], ["skip"], ["skip"], 
            ["ir_k5_e1"], ["skip"], ["skip"], ["skip"], 
            ["ir_k3_e3"], ["skip"], ["skip"], ["skip"], 
            ["skip"], 
            ],
            "block_cfg": {
                "first": [16, 1],
                "stages": [
                    [[1, 16, 1, 1]],                                                        # stage 1
                    [[6, 24, 1, 2]],  [[1, 24, 1, 1]],      [[1, 24, 1, 1]],  [[1, 24, 1, 1]],  # stage 2
                    [[6, 32, 1, 2]],  [[1, 32, 1, 1]],      [[1, 32, 1, 1]],  [[1, 32, 1, 1]],  # stage 3
                    [[6, 64, 1, 2]],  [[1, 64, 1, 1]],      [[1, 64, 1, 1]],  [[1, 64, 1, 1]],  # stage 4
                    [[1, 96, 1, 1]], [[1, 96, 1, 1]],     [[1, 96, 1, 1]], [[1, 96, 1, 1]], # stage 5
                    [[3, 160, 1, 2]], [[1, 160, 1, 1]],     [[1, 160, 1, 1]], [[1, 160, 1, 1]], # stage 6
                    [[1, 320, 1, 1]],                                                       # stage 7
                ],
                "backbone": [num for num in range(23)],
            },
        },

    "1123test1_tmp": {
            "block_op_type": [
            ["ir_k3_e6"], 
            ["ir_k3_e6"], ["skip"], ["ir_k5_e6"], ["ir_k5_e6"], 
            ["ir_k5_e3"], ["ir_k5_e6"], ["ir_k5_e3"], ["ir_k5_e6"], 
            ["ir_k3_e6"], ["ir_k5_e6"], ["ir_k5_e3"], ["skip"], 
            ["ir_k3_e6"], ["skip"], ["ir_k3_e3"], ["skip"], 
            ["ir_k3_e6"], ["skip"], ["ir_k3_e3"], ["ir_k3_e3"], 
            ["ir_k5_e1"], 
            ],
            "block_cfg": {
                "first": [16, 1],
                "stages": [
                    [[6, 16, 1, 1]],                                                        # stage 1
                    [[6, 24, 1, 2]],  [[1, 24, 1, 1]],      [[6, 24, 1, 1]],  [[6, 24, 1, 1]],  # stage 2
                    [[3, 32, 1, 2]],  [[6, 32, 1, 1]],      [[3, 32, 1, 1]],  [[6, 32, 1, 1]],  # stage 3
                    [[6, 64, 1, 2]],  [[6, 64, 1, 1]],      [[3, 64, 1, 1]],  [[1, 64, 1, 1]],  # stage 4
                    [[6, 96, 1, 1]], [[1, 96, 1, 1]],     [[3, 96, 1, 1]], [[1, 96, 1, 1]], # stage 5
                    [[6, 160, 1, 2]], [[1, 160, 1, 1]],     [[3, 160, 1, 1]], [[3, 160, 1, 1]], # stage 6
                    [[1, 320, 1, 1]],                                                       # stage 7
                ],
                "backbone": [num for num in range(23)],
            },
        },
}
