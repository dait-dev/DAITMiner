// See https://aka.ms/new-console-template for more information
// Create main context
using ILGPU;
using ILGPU.IR.Values;
using ILGPU.Runtime;

#region Accelerated algorithm

/// <summary>
/// Multiplies two dense matrices and returns the resultant matrix.
/// </summary>
/// <param name="accelerator">The Accelerator to run the multiplication on</param>
/// <param name="a">A dense MxK matrix</param>
/// <param name="b">A dense KxN matrix</param>
/// <returns>A dense MxN matrix</returns>
static float[,] MatrixMultiplyAccelerated(Accelerator accelerator, float[,] a, float[,] b)
{
    var m = a.GetLength(0);
    var ka = a.GetLength(1);
    var kb = b.GetLength(0);
    var n = b.GetLength(1);

    if (ka != kb)
        throw new ArgumentException($"Cannot multiply {m}x{ka} matrix by {n}x{kb} matrix", nameof(b));

    var kernel = accelerator.LoadAutoGroupedStreamKernel<
        Index2D,
        ArrayView2D<float, Stride2D.DenseX>,
        ArrayView2D<float, Stride2D.DenseX>,
        ArrayView2D<float, Stride2D.DenseX>>(
        MatrixMultiplyAcceleratedKernel);

    using var aBuffer = accelerator.Allocate2DDenseX<float>(new Index2D(m, ka));
    using var bBuffer = accelerator.Allocate2DDenseX<float>(new Index2D(ka, n));
    using var cBuffer = accelerator.Allocate2DDenseX<float>(new Index2D(m, n));
    aBuffer.CopyFromCPU(a);
    bBuffer.CopyFromCPU(b);

    kernel(cBuffer.Extent.ToIntIndex(), aBuffer.View, bBuffer.View, cBuffer.View);

    // Reads data from the GPU buffer into a new CPU array.
    // Implicitly calls accelerator.DefaultStream.Synchronize() to ensure
    // that the kernel and memory copy are completed first.
    return cBuffer.GetAsArray2D();
}

/// <summary>
/// The matrix multiplication kernel that runs on the accelerated device.
/// </summary>
/// <param name="index">Current matrix index</param>
/// <param name="aView">An input matrix of size MxK</param>
/// <param name="bView">An input matrix of size KxN</param>
/// <param name="cView">An output matrix of size MxN</param>
static void MatrixMultiplyAcceleratedKernel(
    Index2D index,
    ArrayView2D<float, Stride2D.DenseX> aView,
    ArrayView2D<float, Stride2D.DenseX> bView,
    ArrayView2D<float, Stride2D.DenseX> cView)
{
    var x = index.X;
    var y = index.Y;
    var sum = 0.0f;

    for (var i = 0; i < aView.IntExtent.Y; i++)
        sum += aView[new Index2D(x, i)] * bView[new Index2D(i, y)];

    cView[index] = sum;
}

#endregion

#region Check version
string version = "0.0.1a";
try
{
    HttpClient cli = new HttpClient();
    HttpResponseMessage resp = cli.GetAsync("https://miningpool.dait.dev/GetLatestVersion.ashx").Result;
    string latestVersion = resp.Content.ReadAsStringAsync().Result;
    if (version != latestVersion)
    {
        Console.WriteLine($"Version {version}, latest version: {latestVersion}. Please update DAITCore application.");
        Console.ReadKey();
        return;
    }
}
catch
{
    return;
}
#endregion


static void PrintAcceleratorInfo(Accelerator accelerator)
{
    Console.WriteLine($"Name: {accelerator.Name}");
    Console.WriteLine($"MemorySize: {accelerator.MemorySize}");
    Console.WriteLine($"MaxThreadsPerGroup: {accelerator.MaxNumThreadsPerGroup}");
    Console.WriteLine($"MaxSharedMemoryPerGroup: {accelerator.MaxSharedMemoryPerGroup}");
    Console.WriteLine($"MaxGridSize: {accelerator.MaxGridSize}");
    Console.WriteLine($"MaxConstantMemory: {accelerator.MaxConstantMemory}");
    Console.WriteLine($"WarpSize: {accelerator.WarpSize}");
    Console.WriteLine($"NumMultiprocessors: {accelerator.NumMultiprocessors}");
}

string minerPubKey = "";
if (!File.Exists("PubKey.txt"))
{
    Console.WriteLine("Enter your solana public key: ");
    minerPubKey = Console.ReadLine();
    File.WriteAllText("PubKey.txt", minerPubKey);
}

using (var context = Context.CreateDefault())
{
    // For each available device...
    foreach (var device in context)
    {
        // Create accelerator for the given device.
        // Note that all accelerators have to be disposed before the global context is disposed
        using var accelerator = device.CreateAccelerator(context);
        Console.WriteLine($"Accelerator: {device.AcceleratorType}, {accelerator.Name}");
        PrintAcceleratorInfo(accelerator);
        Console.WriteLine();
    }
}

