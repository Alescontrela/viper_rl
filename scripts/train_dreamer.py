import importlib
import pathlib
import sys
import warnings
from functools import partial as bind

warnings.filterwarnings('ignore', '.*box bound precision lowered.*')
warnings.filterwarnings('ignore', '.*using stateful random seeds*')
warnings.filterwarnings('ignore', '.*is a deprecated alias for.*')
warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')

directory = pathlib.Path(__file__).resolve()
directory = directory.parent
sys.path.append(str(directory.parent))

from viper_rl.dreamerv3 import embodied
from viper_rl.dreamerv3.embodied import wrappers


def main(argv=None):
  from viper_rl.dreamerv3 import agent as agt

  parsed, other = embodied.Flags(configs=['defaults']).parse_known(argv)
  config = embodied.Config(agt.Agent.configs['defaults'])
  for name in parsed.configs:
    config = config.update(agt.Agent.configs[name])
  config = embodied.Flags(config).parse(other)
  args = embodied.Config(
      **config.run, logdir=config.logdir,
      batch_steps=config.batch_size * config.batch_length)
  print(config)

  logdir = embodied.Path(args.logdir)
  logdir.mkdirs()
  config.save(logdir / 'config.yaml')
  step = embodied.Counter()
  logger = make_logger(parsed, logdir, step, config)

  if config.reward_model != 'none':
      print(f'Loading reward model {config.reward_model}')
      from viper_rl.videogpt.reward_models import LOAD_REWARD_MODEL_DICT
      reward_model = LOAD_REWARD_MODEL_DICT[config.reward_model](
        task=config.task,
        compute_joint=config.reward_model_compute_joint,
        minibatch_size=config.reward_model_batch_size,
        encoding_minibatch_size=config.reward_model_batch_size,
        reward_model_device=config.jax.reward_model_device)
  else:
      reward_model = None

  replay_kwargs = {'reward_model': reward_model}

  cleanup = []
  try:
    if args.script == 'train':
      env = make_envs(config)
      replay = make_replay(config, logdir / 'replay', **replay_kwargs)
      cleanup.append(env)
      agent = agt.Agent(env.obs_space, env.act_space, step, config)
      embodied.run.train(agent, env, replay, logger, args)
  
    elif args.script == 'train_amp':
      reference_replay = make_replay(config, config.reference_dir, is_eval=False, **replay_kwargs)
      print(f'Loaded reference data: {reference_replay.stats}')
      replay = make_replay(config, logdir / 'replay')
      env = make_envs(config)
      cleanup.append(env)
      agent = agt.Agent(env.obs_space, env.act_space, step, config)
      embodied.run.train_amp(agent, env, replay, reference_replay, logger, args)

    elif args.script == 'train_save':
      env = make_envs(config)
      replay = make_replay(config, logdir / 'replay', **replay_kwargs)
      cleanup.append(env)
      agent = agt.Agent(env.obs_space, env.act_space, step, config)
      embodied.run.train_save(agent, env, replay, logger, args)

    elif args.script == 'train_eval':
      env = make_envs(config)
      replay = make_replay(config, logdir / 'replay', **replay_kwargs)
      eval_replay = make_replay(config, logdir / 'eval_replay', is_eval=True)
      eval_env = make_envs(config)  # mode='eval'
      cleanup += [env, eval_env]
      agent = agt.Agent(env.obs_space, env.act_space, step, config)
      embodied.run.train_eval(
          agent, env, eval_env, replay, eval_replay, logger, args)

    elif args.script == 'train_holdout':
      env = make_envs(config)
      replay = make_replay(config, logdir / 'replay', **replay_kwargs)
      if config.eval_dir:
        assert not config.train.eval_fill
        eval_replay = make_replay(config, config.eval_dir, is_eval=True)
      else:
        assert 0 < args.eval_fill <= config.replay_size // 10, args.eval_fill
        eval_replay = make_replay(config, logdir / 'eval_replay', is_eval=True, **replay_kwargs)
      cleanup.append(env)
      agent = agt.Agent(env.obs_space, env.act_space, step, config)
      embodied.run.train_holdout(
          agent, env, replay, eval_replay, logger, args)

    elif args.script == 'eval_only':
      env = make_envs(config)  # mode='eval'
      cleanup.append(env)
      agent = agt.Agent(env.obs_space, env.act_space, step, config)
      embodied.run.eval_only(agent, env, logger, args)

    elif args.script == 'eval_only_save':
      env = make_envs(config)
      agent = agt.Agent(env.obs_space, env.act_space, step, config)
      cleanup.append(env)
      embodied.run.eval_only_save(agent, env, logger, args)

    elif args.script == 'parallel':
      assert config.run.actor_batch <= config.envs.amount, (
          config.run.actor_batch, config.envs.amount)
      step = embodied.Counter()
      env = make_env(config)
      env.close()
      replay = make_replay(config, logdir / 'replay', reward_model=reward_model, rate_limit=True, **replay_kwargs)
      agent = agt.Agent(env.obs_space, env.act_space, step, reward_model, config)
      embodied.run.parallel(
          agent, replay, logger, bind(make_env, config),
          num_envs=config.envs.amount, args=args)

    else:
      raise NotImplementedError(args.script)
  finally:
    for obj in cleanup:
      obj.close()


def make_logger(parsed, logdir, step, config):
  multiplier = config.env.get(config.task.split('_')[0], {}).get('repeat', 1)
  logger = embodied.Logger(step, [
      embodied.logger.TerminalOutput(config.filter),
      embodied.logger.JSONLOutput(logdir, 'metrics.jsonl'),
      embodied.logger.JSONLOutput(logdir, 'scores.jsonl', 'episode/score'),
      embodied.logger.TensorBoardOutput(logdir),
      # embodied.logger.WandBOutput(logdir.name, config),
      # embodied.logger.MLFlowOutput(logdir.name),
  ], multiplier)
  return logger


def make_replay(
    config, directory=None, is_eval=False, rate_limit=False, reward_model=None, **kwargs):
  assert config.replay == 'uniform' or config.replay == 'uniform_relabel' or not rate_limit
  length = config.batch_length
  size = config.replay_size // 10 if is_eval else config.replay_size
  if config.replay == 'uniform_relabel':
    kw = {'online': config.replay_online}
    if rate_limit and config.run.train_ratio > 0:
      kw['samples_per_insert'] = config.run.train_ratio / config.batch_length
      kw['tolerance'] = 10 * config.batch_size
      kw['min_size'] = config.batch_size
    assert reward_model is not None, 'relabel requires reward model'
    replay = embodied.replay.UniformRelabel(
        length, reward_model, config.uniform_relabel_add_mode, size, directory, **kw)
  elif config.replay == 'uniform' or is_eval:
    kw = {'online': config.replay_online}
    if rate_limit and config.run.train_ratio > 0:
      kw['samples_per_insert'] = config.run.train_ratio / config.batch_length
      kw['tolerance'] = 10 * config.batch_size
      kw['min_size'] = config.batch_size
    replay = embodied.replay.Uniform(length, size, directory, **kw)
  elif config.replay == 'reverb':
    replay = embodied.replay.Reverb(length, size, directory)
  elif config.replay == 'chunks':
    replay = embodied.replay.NaiveChunks(length, size, directory)
  else:
    raise NotImplementedError(config.replay)
  return replay


def make_envs(config, **overrides):
  suite, task = config.task.split('_', 1)
  ctors = []
  for index in range(config.envs.amount):
    ctor = lambda: make_env(config, **overrides)
    if config.envs.parallel != 'none':
      ctor = bind(embodied.Parallel, ctor, config.envs.parallel)
    if config.envs.restart:
      ctor = bind(wrappers.RestartOnException, ctor)
    ctors.append(ctor)
  envs = [ctor() for ctor in ctors]
  return embodied.BatchEnv(envs, parallel=(config.envs.parallel != 'none'))


def make_env(config, **overrides):
  # You can add custom environments by creating and returning the environment
  # instance here. Environments with different interfaces can be converted
  # using `embodied.envs.from_gym.FromGym` and `embodied.envs.from_dm.FromDM`.
  suite, task = config.task.split('_', 1)
  ctor = {
      'dummy': 'embodied.envs.dummy:Dummy',
      'gym': 'embodied.envs.from_gym:FromGym',
      'dm': 'embodied.envs.from_dmenv:FromDM',
      'crafter': 'embodied.envs.crafter:Crafter',
      'dmc': 'embodied.envs.dmc:DMC',
      'rlbench': 'embodied.envs.rlbench:RLBench',
      'dmcmulticam': 'embodied.envs.dmcmulticam:DMCMultiCam',
      'atari': 'embodied.envs.atari:Atari',
      'dmlab': 'embodied.envs.dmlab:DMLab',
      'minecraft': 'embodied.envs.minecraft:Minecraft',
      'loconav': 'embodied.envs.loconav:LocoNav',
      'pinpad': 'embodied.envs.pinpad:PinPad',
      'kitchen': 'embodied.envs.kitchen:Kitchen',
      'cliport': 'embodied.envs.cliport:Cliport',
  }[suite]
  if isinstance(ctor, str):
    module, cls = ctor.split(':')
    module = importlib.import_module(module)
    ctor = getattr(module, cls)
  kwargs = config.env.get(suite, {})
  kwargs.update(overrides)
  env = ctor(task, **kwargs)
  return wrap_env(env, config)


def wrap_env(env, config):
  args = config.wrapper
  for name, space in env.act_space.items():
    if name == 'reset':
      continue
    elif space.discrete:
      env = wrappers.OneHotAction(env, name)
    elif args.discretize:
      env = wrappers.DiscretizeAction(env, name, args.discretize)
    else:
      env = wrappers.NormalizeAction(env, name)
  if args.density:
    env = wrappers.Density(env)
  env = wrappers.FlattenTwoDimObs(env)
  env = wrappers.ExpandScalars(env)
  if args.length:
    env = wrappers.TimeLimit(env, args.length, args.reset)
  if args.checks:
    env = wrappers.CheckSpaces(env)
  for name, space in env.act_space.items():
    if not space.discrete:
      env = wrappers.ClipAction(env, name)
  return env


if __name__ == '__main__':
  main()
