# Dataset analysis

Source folder: `datasets\raw\construction-ppe`

Object size = box area as a fraction of image area (small < 0.005, medium < 0.05). Brightness = mean gray level (dark < 60, bright > 180). These are heuristics for error analysis.

## Split: train

- images: 800 (unreadable: 0, without objects: 0)
- distinct resolutions: 9
- objects per image: mean 14.88, max 76
- brightness: {'dark': 3, 'normal': 797, 'bright': 0}

| class | instances | images | small | medium | large | note |
|---|---|---|---|---|---|---|
| helmet | 1282 | 613 | 691 | 578 | 13 |  |
| gloves | 1548 | 546 | 932 | 612 | 4 |  |
| vest | 1476 | 644 | 642 | 345 | 489 |  |
| boots | 2003 | 576 | 1082 | 917 | 4 |  |
| goggles | 612 | 437 | 378 | 233 | 1 |  |
| Person | 1962 | 791 | 611 | 455 | 896 |  |
| no_helmet | 634 | 337 | 427 | 198 | 9 |  |
| no_goggle | 265 | 227 | 264 | 1 | 0 |  |
| no_gloves | 839 | 340 | 665 | 173 | 1 |  |
| no_boots | 569 | 244 | 279 | 287 | 3 |  |
| forklift | 275 | 201 | 13 | 69 | 193 |  |
| no_vest | 437 | 300 | 179 | 94 | 164 |  |

## Split: val

- images: 200 (unreadable: 0, without objects: 0)
- distinct resolutions: 9
- objects per image: mean 14.8, max 61
- brightness: {'dark': 0, 'normal': 199, 'bright': 1}

| class | instances | images | small | medium | large | note |
|---|---|---|---|---|---|---|
| helmet | 294 | 146 | 160 | 132 | 2 |  |
| gloves | 381 | 136 | 245 | 136 | 0 |  |
| vest | 340 | 157 | 154 | 75 | 111 |  |
| boots | 481 | 155 | 254 | 227 | 0 |  |
| goggles | 139 | 102 | 83 | 56 | 0 |  |
| Person | 481 | 200 | 145 | 114 | 222 |  |
| no_helmet | 179 | 91 | 111 | 65 | 3 |  |
| no_goggle | 82 | 56 | 82 | 0 | 0 |  |
| no_gloves | 203 | 94 | 162 | 41 | 0 |  |
| no_boots | 178 | 71 | 110 | 68 | 0 |  |
| forklift | 73 | 46 | 4 | 25 | 44 |  |
| no_vest | 130 | 84 | 53 | 34 | 43 |  |
