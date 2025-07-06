
spoof_type_options = [
    "(1) Real face",
    "(2) Photo",
    "(3) Poster",
    "(4) A4-paper",
    "(5) 2D face mask",
    "(6) 2D upper-body mask",
    "(7) 2D region mask",
    "(8) PC screen",
    "(9) Pad screen",
    "(10) Phone screen",
    "(11) 3D mask",
]

illumination_options = [
    "(1) Normal",
    "(2) Strong",
    "(3) Back",
    "(4) Dark",
]

environment_options = [
    "(1) Indoor",
    "(2) Outdoor",
]

camera_quality_options = [
    "(1) Low",
    "(2) Medium",
    "(3) High",
]

spoof_type_questions = (
    "Choose the correct option to the following question: Which type of spoof type is in this image?\n"
    + " ".join(spoof_type_options)
)

illumination_questions = (
    "Choose the correct option to the following question: What is the illumination condition in this image?\n"
    + " ".join(illumination_options)
)

environment_questions = (
    "Choose the correct option to the following question: What is the environment in this image?\n"
    + " ".join(environment_options)
)

camera_quality_questions = (
    "Choose the correct option to the following question: What is the camera quality in this image?\n"
    + " ".join(camera_quality_options)
)


QA = {
    'presentation attack': {
        'question': spoof_type_questions,
        'options': spoof_type_options,
    },
    'illumination condition': {
        'question': illumination_questions,
        'options': illumination_options,
    },
    'enviroment': {
        'question': environment_questions,
        'options': environment_options,
    },
    'camera quality': {
        'question': camera_quality_questions,
        'options': camera_quality_options,
    },
}


# ChatGPTQA = [
#     {
#         "instruction": "Check for natural skin texture with subtle variations and depth, especially around the nose and cheek areas.",
#         "question": "Does the face display realistic skin texture and depth variations?",
#         "answer": ["yes", "no", "no", "no", "no", "no", "no", "no", "no", "no", "no"]
#     },
#     {
#         "instruction": "Analyze for soft shadow gradients on the face that indicate natural lighting.",
#         "question": "Do the shadows appear soft and natural?",
#     },
#     {
#         "instruction": "Check for flat textures and lack of depth around the nose and eye regions, common in printed photos.",
#         "question": "Is the face flat with limited 3D structure around prominent features?",
#     },
#     {
#         "instruction": "Look for uniform lighting across the face without natural shadow variations.",
#         "question": "Are there minimal or uniform shadows on the face?",
#     },
#     {
#         "instruction": "Identify any signs of pixelation or printing artifacts in the face image.",
#         "question": "Is there visible pixelation or printing artifacts?",
#     },
#     {
#         "instruction": "Detect high pixelation and limited detail resolution common in poster images.",
#         "question": "Does the face appear pixelated with low-resolution details?",
#     },
#     {
#         "instruction": "Look for exaggerated flatness with uniform shadows across the face.",
#         "question": "Is the face flat with uniform shadowing typical of a printed surface?",
#     },
#     {
#         "instruction": "Check for unusual reflections or gloss from the printed surface.",
#         "question": "Are there reflections or gloss indicative of a poster?",
#     },
#     {
#         "instruction": "Look for high contrast edges with a lack of depth and realistic skin texture.",
#         "question": "Is the image contrast high with flat texture and minimal depth?",
#     },
#     {
#         "instruction": "Detect lack of natural shadows around facial features.",
#         "question": "Are there limited shadows around the nose, eyes, and chin?",
#     },
#     {
#         "instruction": "Look for signs of edge warping or bending, typical of thin paper.",
#         "question": "Is there any edge warping or bending noticeable?",
#     },
#     {
#         "instruction": "Identify shallow 3D structure, particularly around the cheeks and nose, with flatness in other areas.",
#         "question": "Is there a slight 3D structure but overall flatness on the face?",
#     },
#     {
#         "instruction": "Look for edges of the mask around the face or irregular skin texture.",
#         "question": "Are mask edges visible or is the skin texture inconsistent?",
#     },
#     {
#         "instruction": "Analyze lighting inconsistencies; masks often reflect light differently than real skin.",
#         "question": "Are there inconsistent lighting effects on the face?",
#     },
#     {
#         "instruction": "Check for flatness in the upper body with limited contouring around shoulders and neck.",
#         "question": "Is there a flat appearance around the upper body and neck?",
#     },
#     {
#         "instruction": "Look for unnatural uniformity across the body and face, common with 2D masks.",
#         "question": "Is the texture uniform across the entire upper body?",
#     },
#     {
#         "instruction": "Detect the absence of depth and shadow variations on the upper body.",
#         "question": "Are depth and shadow variations minimal on the upper-body?",
#     },
#     {
#         "instruction": "Focus on individual face regions for signs of masking (e.g., shallow nose or mouth).",
#         "question": "Is there minimal 3D detail in specific regions like the nose or mouth?",
#     },
#     {
#         "instruction": "Look for irregular depth, especially around isolated facial areas like eyes or cheeks.",
#         "question": "Does depth vary unnaturally across specific face regions?",
#     },
#     {
#         "instruction": "Identify region-specific lighting inconsistencies typical of layered masks.",
#         "question": "Are there unnatural lighting patterns on specific face areas?",
#     },
#     {
#         "instruction": "Detect pixelation and moiré patterns often seen on PC screens.",
#         "question": "Are there pixelation or screen patterns visible?",
#     },
#     {
#         "instruction": "Identify flatness and glare commonly observed on large screens.",
#         "question": "Is there flatness with possible screen glare?",
#     },
#     {
#         "instruction": "Look for reflections from the PC screen that differ from natural reflections.",
#         "question": "Are reflections present that appear artificial?",
#     },
#     {
#         "instruction": "Check for smaller pixel sizes and limited depth typical of pad screens.",
#         "question": "Is there a flat appearance with screen pixel patterns?",
#     },
#     {
#         "instruction": "Identify minimal screen glare and flat texture on the face.",
#         "question": "Is the face flat with minimal screen glare?",
#     },
#     {
#         "instruction": "Look for minor reflections or artifacts from the pad’s display.",
#         "question": "Are there minor reflections typical of a pad screen?",
#     },
#     {
#         "instruction": "Detect small pixel patterns and flatness typical of smaller screens.",
#         "question": "Are there small screen pixels visible?",
#     },
#     {
#         "instruction": "Look for minor reflections and flat features indicative of a phone.",
#         "question": "Is the face flat with subtle screen reflections?",
#     },
#     {
#         "instruction": "Check for high pixel density but lack of depth or realistic shadows.",
#         "question": "Is there high pixel density but no depth cues?",
#     },
#     {
#         "instruction": "Identify 3D contours but look for unrealistic skin texture or color transitions.",
#         "question": "Does the face have 3D contours with unnatural skin tones?",
#     },
#     {
#         "instruction": "Check for hard mask edges around the face, particularly near the hairline or jaw.",
#         "question": "Are there visible mask edges around the face?",
#     },
#     {
#         "instruction": "Look for reflection patterns inconsistent with human skin, typical of synthetic materials.",
#         "question": "Are reflection patterns unnatural, suggesting synthetic material?",
#     }
# ]


gptQA = [
    {
        "instruction": "Analyze for soft, natural shadow gradients on the face, typical of real lighting conditions.",
        "question": "Do the shadows appear soft and natural?",
        "answer": ["yes", "no", "no", "no", "no", "no", "no", "no", "no", "no", "no"]
    },
    {
        "instruction": "Check for flat textures and lack of 3D depth around facial features, as in printed photos and screens.",
        "question": "Is the face flat with limited 3D structure around prominent features?",
        "answer": ["no", "yes", "yes", "yes", "yes", "yes", "yes", "yes", "yes", "yes", "maybe"]
    },
    {
        "instruction": "Observe for uniform lighting on the face without natural shadow variations, which may indicate spoofing.",
        "question": "Are there minimal or uniform shadows on the face?",
        "answer": ["no", "yes", "yes", "yes", "yes", "yes", "yes", "yes", "yes", "yes", "maybe"]
    },
    {
        "instruction": "Identify signs of pixelation or printing artifacts, such as moiré patterns, typical in printed and screen-based spoofs.",
        "question": "Is there visible pixelation or printing artifacts?",
        "answer": ["no", "yes", "yes", "yes", "yes", "yes", "yes", "yes", "yes", "yes", "no"]
    },
    {
        "instruction": "Detect low-resolution details and high pixelation levels that may appear in spoofing with posters or low-quality prints.",
        "question": "Does the face appear pixelated with low-resolution details?",
        "answer": ["no", "maybe", "yes", "maybe", "maybe", "maybe", "maybe", "no", "no", "no", "maybe"]
    },
    {
        "instruction": "Look for exaggerated flatness with even shadowing, a feature common in printed images like posters and A4 paper.",
        "question": "Is the face flat with uniform shadowing typical of a printed surface?",
        "answer": ["no", "yes", "yes", "yes", "yes", "yes", "yes", "yes", "yes", "yes", "no"]
    },
    {
        "instruction": "Check for unusual reflections or glossy areas that indicate printed or screen-based surfaces.",
        "question": "Are there reflections or gloss indicative of a poster or screen?",
        "answer": ["no", "yes", "yes", "yes", "maybe", "maybe", "maybe", "yes", "yes", "yes", "no"]
    },
    {
        "instruction": "Look for high-contrast edges with flat textures and minimal depth, as would be seen in printed paper or screen.",
        "question": "Is the image contrast high with flat texture and minimal depth?",
        "answer": ["no", "yes", "yes", "yes", "yes", "yes", "yes", "yes", "yes", "yes", "no"]
    },
    {
        "instruction": "Check for limited or unnatural shadow details around facial features, often present in printed spoofs or screen spoofs.",
        "question": "Are there limited shadows around the nose, eyes, and chin?",
        "answer": ["no", "yes", "yes", "yes", "yes", "yes", "yes", "yes", "yes", "yes", "no"]
    },
    {
        "instruction": "Identify any warping or bending that may be visible along the edges, as with thin paper.",
        "question": "Is there any edge warping or bending noticeable?",
        "answer": ["no", "yes", "yes", "yes", "maybe", "maybe", "maybe", "yes", "yes", "yes", "no"]
    },
    {
        "instruction": "Look for shallow 3D structure in parts of the face, such as around the cheeks, combined with flatness in other areas.",
        "question": "Is there a slight 3D structure but overall flatness on the face?",
        "answer": ["no", "no", "no", "no", "yes", "yes", "yes", "no", "no", "no", "no"]
    },
    {
        "instruction": "Observe for visible mask edges or irregular skin textures, which may indicate a face mask.",
        "question": "Are mask edges visible or is the skin texture inconsistent?",
        "answer": ["no", "no", "no", "no", "yes", "yes", "yes", "no", "no", "no", "yes"]
    },
    {
        "instruction": "Analyze the lighting for inconsistencies that may suggest a mask, as masks often reflect light differently than real skin.",
        "question": "Are there inconsistent lighting effects on the face?",
        "answer": ["no", "no", "no", "no", "yes", "yes", "yes", "no", "no", "no", "yes"]
    },
    {
        "instruction": "Check for a lack of depth and shadow variations on the upper-body of face, characteristic of 2D upper-body masks.",
        "question": "Are depth and shadow variations minimal on the upper-body of face?",
        "answer": ["no", "no", "no", "no", "no", "yes", "no", "no", "no", "no", "no"]
    },
    {
        "instruction": "Check for irregular depth across facial areas like the eyes or cheeks, indicating possible masking.",
        "question": "Does depth vary unnaturally across specific face regions?",
        "answer": ["no", "no", "no", "no", "no", "yes", "yes", "no", "no", "no", "no"]
    },
    {
        "instruction": "Look for unnatural lighting patterns concentrated on specific face areas, as with layered masks.",
        "question": "Are there unnatural lighting patterns on specific face areas?",
        "answer": ["no", "no", "no", "no", "maybe", "yes", "yes", "no", "no", "no", "maybe"]
    },
    {
        "instruction": "Detect signs of pixelation or moiré patterns common on screens.",
        "question": "Are there pixelation or screen patterns visible?",
        "answer": ["no", "no", "no", "no", "no", "no", "no", "yes", "yes", "yes", "no"]
    },
    {
        "instruction": "Observe for flatness combined with screen glare, characteristics often found on screens.",
        "question": "Is there flatness with possible screen glare?",
        "answer": ["no", "no", "no", "no", "no", "no", "no", "yes", "yes", "yes", "no"]
    },
    {
        "instruction": "Check for reflections that look artificial, which may appear on screens.",
        "question": "Are reflections present that appear artificial?",
        "answer": ["no", "no", "no", "no", "no", "no", "no", "yes", "yes", "yes", "no"]
    },
    {
        "instruction": "Identify small pixel sizes and limited depth, typically associated with pad screens.",
        "question": "Is there a flat appearance with screen pixel patterns?",
        "answer": ["no", "no", "no", "no", "no", "no", "no", "no", "yes", "no", "no"]
    },
    {
        "instruction": "Look for a flat texture and minimal screen glare, indicators of a pad screen or phone screen.",
        "question": "Is the face flat with minimal screen glare?",
        "answer": ["no", "no", "no", "no", "no", "no", "no", "yes", "yes", "maybe", "no"]
    },
    {
        "instruction": "Identify 3D contours but look for unrealistic skin texture or color transitions.",
        "question": "Does the face have 3D contours with unnatural skin tones?",
        "answer": ["no", "no", "no", "no", "no", "no", "no", "no", "no", "no", "yes"]
    },
    {
        "instruction": "Check for hard mask edges around the face, particularly near the hairline or jaw.",
        "question": "Are there visible mask edges around the face?",
        "answer": ["no", "no", "no", "no", "yes", "maybe", "maybe", "no", "no", "no", "yes"]
    },
    {
        "instruction": "Look for reflection patterns inconsistent with human skin, typical of synthetic materials.",
        "question": "Are reflection patterns unnatural, suggesting synthetic material?",
        "answer": ["no", "no", "no", "no", "no", "no", "no", "no", "no", "no", "yes"]
    }
]
