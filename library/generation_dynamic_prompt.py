"""Dynamic prompt expansion shared by direct generation and the comparison planner."""
import itertools
import logging
import random
import re

logger=logging.getLogger(__name__)

RE_DYNAMIC_PROMPT = re.compile(r"\{((e|E)\$\$)?(([\d\-]+)\$\$)?(([^\|\}]+?)\$\$)?(.+?((\|).+?)*?)\}")


def handle_dynamic_prompt_variants(prompt, repeat_count, rng=random):
    founds = list(RE_DYNAMIC_PROMPT.finditer(prompt))
    if not founds:
        return [prompt]

    # make each replacement for each variant
    enumerating = False
    replacers = []
    for found in founds:
        # if "e$$" is found, enumerate all variants
        found_enumerating = found.group(2) is not None
        enumerating = enumerating or found_enumerating

        separator = ", " if found.group(6) is None else found.group(6)
        variants = found.group(7).split("|")

        # parse count range
        count_range = found.group(4)
        if count_range is None:
            count_range = [1, 1]
        else:
            count_range = count_range.split("-")
            if len(count_range) == 1:
                count_range = [int(count_range[0]), int(count_range[0])]
            elif len(count_range) == 2:
                count_range = [int(count_range[0]), int(count_range[1])]
            else:
                logger.warning(f"invalid count range: {count_range}")
                count_range = [1, 1]
            if count_range[0] > count_range[1]:
                count_range = [count_range[1], count_range[0]]
            if count_range[0] < 0:
                count_range[0] = 0
            if count_range[1] > len(variants):
                count_range[1] = len(variants)

        if found_enumerating:
            # make function to enumerate all combinations
            def make_replacer_enum(vari, cr, sep):
                def replacer():
                    values = []
                    for count in range(cr[0], cr[1] + 1):
                        for comb in itertools.combinations(vari, count):
                            values.append(sep.join(comb))
                    return values

                return replacer

            replacers.append(make_replacer_enum(variants, count_range, separator))
        else:
            # make function to choose random combinations
            def make_replacer_single(vari, cr, sep):
                def replacer():
                    count = rng.randint(cr[0], cr[1])
                    comb = rng.sample(vari, count)
                    return [sep.join(comb)]

                return replacer

            replacers.append(make_replacer_single(variants, count_range, separator))

    # make each prompt
    if not enumerating:
        # if not enumerating, repeat the prompt, replace each variant randomly
        prompts = []
        for _ in range(repeat_count):
            current = prompt
            for found, replacer in zip(founds, replacers):
                current = current.replace(found.group(0), replacer()[0], 1)
            prompts.append(current)
    else:
        # if enumerating, iterate all combinations for previous prompts
        prompts = [prompt]

        for found, replacer in zip(founds, replacers):
            if found.group(2) is not None:
                # make all combinations for existing prompts
                new_prompts = []
                for current in prompts:
                    replecements = replacer()
                    for replecement in replecements:
                        new_prompts.append(current.replace(found.group(0), replecement, 1))
                prompts = new_prompts

        for found, replacer in zip(founds, replacers):
            # make random selection for existing prompts
            if found.group(2) is None:
                for i in range(len(prompts)):
                    prompts[i] = prompts[i].replace(found.group(0), replacer()[0], 1)

    return prompts
