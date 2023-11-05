# Results

## Linear Solving

#### CNN Architecture

* 3-layer CNN
```
n1 = Sequential(
    Conv2d(kernel_size=3),
    ReLU(),
    Conv2d(kernel_size=3, stride=2),
    ReLU(),
    Conv2d(kernel_size=7, padding=0)
)
```

* 2-layer CNN
```
n2 = Sequential(
    Conv2d(kernel_size=2),
    ReLU(),
    Conv2d(kernel_size=2, padding=0)
)
```



<table>
  <tr>
    <th>CNN Architecture</th>
    <th> Input dimension <br> C * H * W</th>
    <th>Problem Size</th>
    <th>Norm Error</th>
    <th>Duration</th>
  </tr>
  <tr>
    <td rowspan="4">n1</td>
    <td rowspan="4">$3 \times 14 \times 14$</td>
    <td>512</td>
    <td>$2.86 \times10^{-1}$</td>
    <td>00:00:38</td>
  </tr>
  <tr></tr>
    <td>1024</td>
    <td>$5.31 \times 10 ^0$</td>
    <td>00:03:06</td>
  </tr>
</table>
