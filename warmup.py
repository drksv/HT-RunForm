def get_warmup_cooldown(age, gender, experience):
    warmup = []
    cooldown = []

    if experience == "Beginner":
        warmup = [
            "5 min brisk walk",
            "Leg swings",
            "Ankle rotations"
        ]
        cooldown = [
            "Hamstring stretch",
            "Calf stretch"
        ]
    elif experience == "Intermediate":
        warmup = [
            "High knees",
            "Butt kicks",
            "Hip openers"
        ]
        cooldown = [
            "Quad stretch",
            "Hip flexor stretch"
        ]
    else:
        warmup = [
            "Dynamic lunges",
            "Bounding drills",
            "Strides"
        ]
        cooldown = [
            "Foam rolling",
            "Deep hip stretches"
        ]

    if age > 40:
        warmup.append("Extra knee mobility")
        cooldown.append("Joint mobility stretches")

    return {
        "warmup": warmup,
        "cooldown": cooldown
    }
