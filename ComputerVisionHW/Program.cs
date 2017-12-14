using System;
using OpenCvSharp;

namespace ComputerVisionHW
{
    static class Program
    {
        /// <summary>
        /// The main entry point for the application.
        /// </summary>
        [STAThread]
        static void Main()
        {
            Mat mat = new Mat("lenna.png");
            Mat result = mat.EmptyClone();

            Mat matGray = new Mat();
            Cv2.CvtColor(mat, matGray, ColorConversionCodes.BGRA2GRAY);

            Mat edges = new Mat();

            Mat gradX = new Mat();
            Mat gradY = new Mat();
            Mat absGradX = new Mat();
            Mat absGradY = new Mat();

            Cv2.Sobel(matGray, gradX, MatType.CV_16S, 1, 0);
            Cv2.Sobel(matGray, gradY, MatType.CV_16S, 0, 1);

            Cv2.ConvertScaleAbs(gradX, absGradX);
            Cv2.ConvertScaleAbs(gradY, absGradY);

            Mat grad = new Mat();

            Cv2.AddWeighted(absGradX, 1.0, absGradY, 1.0, 0, grad);

            Mat dist = new Mat();

            Cv2.Canny(matGray, edges, 50, 200);

            Cv2.DistanceTransform(1 - edges, dist, DistanceTypes.L2, DistanceMaskSize.Mask3);

            Mat normDist = new Mat();

            Cv2.Normalize(dist, normDist, 0, 1.0, NormTypes.MinMax);

            Mat integralImage = new Mat();

            Cv2.Integral(mat, integralImage, MatType.CV_32F);

            for (int i = 0; i < result.Width; i++)
            {
                for (int j = 0; j < result.Height; j++)
                {
                    int size = (int)(10 * dist.Get<float>(i, j));
                    if (size >= 1)
                    {
                        int pixelsCount = ((Clamp(i + size, 0, integralImage.Width - 1) - Clamp(i - size, 0, integralImage.Width - 1)) *
                    (Clamp(j + size, 0, integralImage.Height - 1) - Clamp(j - size, 0, integralImage.Height - 1)));

                        var p0 = new Point(Clamp(i - size, 0, integralImage.Width - 1), Clamp(j - size, 0, integralImage.Height - 1));
                        var p1 = new Point(Clamp(i + size, 0, integralImage.Width - 1), Clamp(j + size, 0, integralImage.Height - 1));
                        var p2 = new Point(Clamp(i - size, 0, integralImage.Width - 1), Clamp(j + size, 0, integralImage.Height - 1));
                        var p3 = new Point(Clamp(i + size, 0, integralImage.Width - 1), Clamp(j - size, 0, integralImage.Height - 1));

                        result.Set<Vec3b>(i, j, new Vec3b(
                            (byte)((
                    integralImage.Get<Vec3f>(p0.X, p0.Y).Item0
                    + integralImage.Get<Vec3f>(p1.X, p1.Y).Item0
                    - integralImage.Get<Vec3f>(p2.X, p2.Y).Item0
                    - integralImage.Get<Vec3f>(p3.X, p3.Y).Item0
                    ) / pixelsCount),
                    (byte)((
                    integralImage.Get<Vec3f>(p0.X, p0.Y).Item1
                    + integralImage.Get<Vec3f>(p1.X, p1.Y).Item1
                    - integralImage.Get<Vec3f>(p2.X, p2.Y).Item1
                    - integralImage.Get<Vec3f>(p3.X, p3.Y).Item1
                    ) / pixelsCount),
                    (byte)((
                    integralImage.Get<Vec3f>(p0.X, p0.Y).Item2
                    + integralImage.Get<Vec3f>(p1.X, p1.Y).Item2
                    - integralImage.Get<Vec3f>(p2.X, p2.Y).Item2
                    - integralImage.Get<Vec3f>(p3.X, p3.Y).Item2
                    ) / pixelsCount)));
                    }
                    else
                        result.Set<Vec3b>(i, j, mat.Get<Vec3b>(i, j));
                }
            }
            using (new Window("src image", mat))
            using (new Window("matGray image", matGray))
            using (new Window("grad image", grad))
            using (new Window("edges image", edges))
            using (new Window("normDist image", normDist))
            using (new Window("result image", result))
            {
                Cv2.WaitKey();
            }

        }

        public static int Clamp(int value, int min, int max)
        {
            return (value < min) ? min : (value > max) ? max : value;
        }

        static double Distance(int x, int y, int x1, int y1)
        {
            return Math.Sqrt((x1 - x) * (x1 - x) + (y1 - y) * (y1 - y));
        }
    }
}