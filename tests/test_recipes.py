"""Pixel-head recipes: table values and the fallback rule."""

import pytest

import config as c
from core.recipes import dataset_identity, resolve_recipe

INSPLAD = r'C:\datasets\insplad-seg\insplad-seg'

# Original best_per_class table (resolve_training_recipe before the refactoring):
# (feat_norm_levels, clamp, pad_mode, tta, score_mode, levels, align_corners, sigma, image_score)
BEST_PER_CLASS = {
    'glass-insulator': ('012', 0.5, 'reflect', 'flips', 'per_level_std', '02', False, 2.0, 'topk'),
    'lightning-rod-suspension': (None, 1.9, 'replicate', 'flips', 'cflow_original', '012', True, 8.0, 'topk'),
    'polymer-insulator-upper-shackle': ('012', 1.9, 'replicate', 'hflip', 'per_level_std', '012', False, 4.0, 'max'),
    'vari-grip': ('012', 1.9, 'reflect', 'hflip', 'cflow_original', '0', True, 8.0, 'topk'),
    'yoke-suspension': ('0', 0.5, 'reflect', 'flips', 'per_level_std', '02', True, 12.0, 'topk'),
}
GENERAL_LEVELS = {'glass-insulator': '02', 'lightning-rod-suspension': '012',
                  'polymer-insulator-upper-shackle': '12', 'vari-grip': '0', 'yoke-suspension': '012'}
EPOCHS = {'glass-insulator': (40, 5), 'lightning-rod-suspension': (40, 5),
          'polymer-insulator-upper-shackle': (30, 3), 'vari-grip': (40, 5), 'yoke-suspension': (12, 2)}


@pytest.mark.parametrize('class_name', sorted(BEST_PER_CLASS))
def test_best_per_class_matches_original_table(class_name):
    r = resolve_recipe(INSPLAD, class_name, 'best_per_class')
    norm, clamp, pad_mode, tta, score, levels, align, sigma, image_score = BEST_PER_CLASS[class_name]
    assert r['planned'] and r['name'] == 'best_per_class' and not r['level_ablation']
    assert (r['training']['feat_norm_levels'], r['training']['clamp_scale']) == (norm, clamp)
    assert r['training']['levels'] == '012'
    inf = r['inference']
    assert (inf['pad'], inf['pad_mode'], inf['tta'], inf['score_mode'], inf['levels'],
            inf['align_corners'], inf['sigma'], inf['image_score']) == \
        (112, pad_mode, tta, score, levels, align, sigma, image_score)
    assert (r['epochs'], r['eval_interval']) == EPOCHS[class_name]


@pytest.mark.parametrize('class_name', sorted(GENERAL_LEVELS))
def test_general_recipe(class_name):
    r = resolve_recipe(INSPLAD, class_name, 'general')
    inf = r['inference']
    assert r['training'] == dict(levels='012', feat_norm_levels=None, clamp_scale=0.5)
    assert (inf['pad'], inf['pad_mode'], inf['tta'], inf['score_mode'], inf['align_corners'],
            inf['sigma'], inf['image_score'], inf['levels']) == \
        (112, 'reflect', 'flips', 'per_level_std', True, 8.0, 'max', GENERAL_LEVELS[class_name])


def test_baseline_recipe():
    inf = resolve_recipe(INSPLAD, 'vari-grip', 'baseline')['inference']
    assert (inf['pad'], inf['tta'], inf['score_mode'], inf['levels'], inf['sigma']) == \
        (0, 'none', 'raw', '012', 4.0)


def test_default_recipe_is_best_per_class():
    assert c.pixel_recipe == 'best_per_class'
    assert resolve_recipe(INSPLAD, 'vari-grip')['name'] == 'best_per_class'


@pytest.mark.parametrize('dataset, class_name', [
    (r'C:\datasets\insplad-seg-no-back-treino', 'vari-grip'),   # another dataset
    (INSPLAD, 'new-class'),                                   # class without a recipe
])
def test_fallback_uses_all_levels_and_level_ablation(dataset, class_name):
    r = resolve_recipe(dataset, class_name)
    assert r['name'] == 'fallback' and not r['planned'] and r['level_ablation']
    assert r['training']['levels'] == '012' and r['inference']['levels'] == '012'
    assert r['inference']['score_mode'] == 'per_level_std'
    assert (r['epochs'], r['eval_interval']) == c.FALLBACK_EPOCHS


def test_level_ablation_override():
    assert resolve_recipe(INSPLAD, 'vari-grip', level_ablation='always')['level_ablation']
    assert not resolve_recipe('x/outro', 'vari-grip', level_ablation='never')['level_ablation']


def test_dataset_identity_ignores_trailing_separator():
    assert dataset_identity(INSPLAD + '\\') == 'insplad-seg'
