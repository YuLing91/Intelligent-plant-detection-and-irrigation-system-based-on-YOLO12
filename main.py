

from DRL.get_features import getFeature
from DRL.training import train
from common.configUtils import getConfig

configFilePath = 'DRL/config/CPU.yaml'

cfg = getConfig(configFilePath)
print("Load Config from: %s" % configFilePath)

if __name__ == '__main__':
    # getFeature(cfg)
    train(cfg)
