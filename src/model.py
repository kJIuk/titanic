from torch import nn


class Model(nn.Module):
    def __init__(self, input_size):
        super(Model, self).__init__()
        modifiers = [1, 12]

        body = []
        for i, mod in enumerate(modifiers[:-1]):
            body.append(nn.Linear(mod * input_size, modifiers[i + 1] * input_size))
            body.append(nn.ReLU())
            body.append(nn.BatchNorm1d(modifiers[i + 1] * input_size))
            # body.append(nn.Dropout())

        self.body = nn.Sequential(*body)

        self.classifier = nn.Linear(modifiers[-1] * input_size, 1)
        self.prob = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, input_t):

        x = self.body(input_t)
        x = self.classifier(x)

        res = self.prob(x)

        return res
