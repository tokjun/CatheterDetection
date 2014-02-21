/*=========================================================================

  Program:   3D Slicer Needle Detection CLI
  Language:  C++
  Author:    Junichi Tokuda, Ph.D. (Brigham and Women's Hospital)

  Copyright (c) Brigham and Women's Hospital. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/

#if defined(_MSC_VER)
#pragma warning ( disable : 4786 )
#endif

#ifdef __BORLANDC__
#define ITK_LEAN_AND_MEAN
#endif

#include "itkHessianRecursiveGaussianImageFilter.h"
#include "itkSmoothingRecursiveGaussianImageFilter.h"
//#include "itkHessian3DToNeedleImageFilter.h"

#include "itkSymmetricSecondRankTensor.h"
#include "itkLabelToNeedleImageFilter.h"

#include "itkOtsuThresholdImageFilter.h"
#include "itkConnectedComponentImageFilter.h"
#include "itkRelabelComponentImageFilter.h"
#include "itkMinimumMaximumImageFilter.h"

//#include "itkMultiplyByConstantImageFilter.h"
#include "itkTransformFileWriter.h"

//#include "itkHessian3DToVesselnessMeasureImageFilter.h"
#include "itkHessianToObjectnessMeasureImageFilter.h"

#include "itkMultiScaleHessianBasedMeasureImageFilter.h"
#include "itkRescaleIntensityImageFilter.h"


#include "itkLabelStatisticsImageFilter.h"
#include "itkBinaryImageToLabelMapFilter.h"

#include "itkChangeLabelImageFilter.h"

#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkCastImageFilter.h"


#include "itkPluginUtilities.h"
#include "CatheterDetectionCLP.h"


// Use an anonymous namespace to keep class types and function names
// from colliding when module is used as shared object module.  Every
// thing should be in an anonymous namespace except for the module
// entry point, e.g. main()
//
namespace {

template<class T> int DoIt( int argc, char * argv[], T )
{
  PARSE_ARGS;

  const     unsigned int        Dimension       = 3;

  typedef   T                   FileInputPixelType;
  typedef   float               InternalPixelType;
  typedef   int                 OutputPixelType;

  typedef   itk::Image< FileInputPixelType, Dimension > FileInputImageType;
  typedef   itk::Image< InternalPixelType, Dimension >  InternalImageType;
  typedef   itk::Image< OutputPixelType, Dimension >    OutputImageType;

  typedef   itk::ImageFileReader< InternalImageType >  ReaderType;
  //typedef   itk::ImageFileWriter< InternalImageType > WriterType;

  // Smoothing filter
  typedef   itk::SmoothingRecursiveGaussianImageFilter<
    InternalImageType, InternalImageType > SmoothingFilterType;
  
  // Line enhancement filter
  //typedef   itk::Hessian3DToNeedleImageFilter<
  //  InternalPixelType > LineFilterType;
  // Otsu Threshold Segmentation filter
  typedef   itk::OtsuThresholdImageFilter<
    InternalImageType, InternalImageType >  OtsuFilterType;
  typedef   itk::ConnectedComponentImageFilter<
    InternalImageType, OutputImageType >  CCFilterType;
  typedef   itk::RelabelComponentImageFilter<
    OutputImageType, OutputImageType > RelabelType;
  // Line detection filter
  typedef   itk::LabelToNeedleImageFilter<
    OutputImageType, OutputImageType > NeedleFilterType;

  // Declare the type of enhancement filter - use ITK's 3D vesselness (Sato)
  // Declare the type of multiscale enhancement filter
  typedef itk::RescaleIntensityImageFilter<InternalImageType> RescaleFilterType;
  //typedef itk::Hessian3DToVesselnessMeasureImageFilter<double> LineFilterType;

  typedef itk::NumericTraits< InternalPixelType >::RealType RealPixelType;
  typedef itk::SymmetricSecondRankTensor< RealPixelType, Dimension > HessianPixelType;
  typedef itk::Image< HessianPixelType, Dimension >                  HessianImageType;
  typedef itk::MultiScaleHessianBasedMeasureImageFilter< InternalImageType, HessianImageType, InternalImageType >
    MultiScaleEnhancementFilterType;
  typedef itk::HessianToObjectnessMeasureImageFilter< HessianImageType, InternalImageType > ObjectnessFilterType;


  typename ReaderType::Pointer reader = ReaderType::New();  

  typename SmoothingFilterType::Pointer smoothing = SmoothingFilterType::New();

  /*
  typename HessianFilterType::Pointer hessianFilter = HessianFilterType::New();
  */
  //typename LineFilterType::Pointer lineFilter = LineFilterType::New();
  
  typename ObjectnessFilterType::Pointer objectnessFilter = ObjectnessFilterType::New();
  MultiScaleEnhancementFilterType::Pointer multiScaleEnhancementFilter = MultiScaleEnhancementFilterType::New();
  typename OtsuFilterType::Pointer OtsuFilter = OtsuFilterType::New();
  typename CCFilterType::Pointer CCFilter = CCFilterType::New();
  typename RelabelType::Pointer RelabelFilter = RelabelType::New();
  typename NeedleFilterType::Pointer needleFilter = NeedleFilterType::New();

  reader->SetFileName( inputVolume.c_str() );

  smoothing->SetInput( reader->GetOutput() );
  smoothing->SetSigma( static_cast< double >(sigma1) );

  //lineFilter->SetPositiveContrast(positivecontrast);
  //lineFilter->SetMinimumLineMeasure(minlinemeasure);
  //lineFilter->SetAlpha1( static_cast< double >(alpha1));
  //lineFilter->SetAlpha2( static_cast< double >(alpha2));
  //lineFilter->SetAngleThreshold (static_cast< double >(anglethreshold) );
  //lineFilter->SetNormal (static_cast< double >(normal[0]),
  //                       static_cast< double >(normal[1]),
  //                       static_cast< double >(normal[2]));

  objectnessFilter->SetBrightObject( positivecontrast );
  objectnessFilter->SetScaleObjectnessMeasure( false );
  objectnessFilter->SetAlpha( alpha );
  objectnessFilter->SetBeta( beta );
  objectnessFilter->SetGamma( gamma );

  multiScaleEnhancementFilter->SetInput(smoothing->GetOutput());
  multiScaleEnhancementFilter->SetSigmaMinimum(minsigma);
  multiScaleEnhancementFilter->SetSigmaMaximum(maxsigma);
  multiScaleEnhancementFilter->SetNumberOfSigmaSteps(stepsigma);
  multiScaleEnhancementFilter->SetHessianToMeasureFilter (objectnessFilter);

  OtsuFilter->SetInput( multiScaleEnhancementFilter->GetOutput());
  OtsuFilter->SetOutsideValue( 255 );
  OtsuFilter->SetInsideValue(  0  );
  OtsuFilter->SetNumberOfHistogramBins( numberOfBins );
  CCFilter->SetInput (OtsuFilter->GetOutput());
  CCFilter->FullyConnectedOff();

  RelabelFilter->SetInput ( CCFilter->GetOutput() );
  RelabelFilter->SetMinimumObjectSize( minimumObjectSize );
  RelabelFilter->Update();

  // Label statistics to pick up the best tubular structure
  typedef itk::LabelStatisticsImageFilter< InternalImageType, OutputImageType > LabelStatisticsImageFilterType;
  typename LabelStatisticsImageFilterType::Pointer labelStatisticsImageFilter = LabelStatisticsImageFilterType::New();
  labelStatisticsImageFilter->SetLabelInput( RelabelFilter->GetInput() );
  labelStatisticsImageFilter->SetInput( multiScaleEnhancementFilter->GetInput() );
  labelStatisticsImageFilter->Update();
  std::cout << "Number of labels: " << labelStatisticsImageFilter->GetNumberOfLabels() << std::endl;
  std::cout << std::endl;  

  typedef LabelStatisticsImageFilterType::ValidLabelValuesContainerType ValidLabelValuesType;
  typedef LabelStatisticsImageFilterType::LabelPixelType                LabelPixelType;

  typedef std::vector< LabelPixelType > LabelListType;
  LabelListType labelsToRemove;
  labelsToRemove.clear();

  for(ValidLabelValuesType::const_iterator vIt=labelStatisticsImageFilter->GetValidLabelValues().begin();
      vIt != labelStatisticsImageFilter->GetValidLabelValues().end();
      ++vIt)
    {
    if ( labelStatisticsImageFilter->HasLabel(*vIt) )
      {
      LabelPixelType labelValue = *vIt;
      //std::cout << "min: " << labelStatisticsImageFilter->GetMinimum( labelValue ) << std::endl;
      //std::cout << "max: " << labelStatisticsImageFilter->GetMaximum( labelValue ) << std::endl;
      //std::cout << "median: " << labelStatisticsImageFilter->GetMedian( labelValue ) << std::endl;
      //std::cout << "mean: " << labelStatisticsImageFilter->GetMean( labelValue ) << std::endl;
      //std::cout << "sigma: " << labelStatisticsImageFilter->GetSigma( labelValue ) << std::endl;
      //std::cout << "variance: " << labelStatisticsImageFilter->GetVariance( labelValue ) << std::endl;
      //std::cout << "sum: " << labelStatisticsImageFilter->GetSum( labelValue ) << std::endl;
      //std::cout << "count: " << labelStatisticsImageFilter->GetCount( labelValue ) << std::endl;
      //std::cout << "region: " << labelStatisticsImageFilter->GetRegion( labelValue ) << std::endl;
      if ( labelStatisticsImageFilter->GetMean( labelValue ) < minimumMeanObjectnessMeasure )
        {
        labelsToRemove.push_back(labelValue);
        }
      }
    }

  //typedef itk::BinaryImageToLabelMapFilter<OutputImageType> BinaryImageToLabelMapFilterType;
  //BinaryImageToLabelMapFilterType::Pointer BinaryFilter = BinaryImageToLabelMapFilterType::New();
  //BinaryFilter->SetInput( RelabelFilter->GetOutput() );
  //BinaryFilter->Update();
  typedef itk::ChangeLabelImageFilter< OutputImageType, OutputImageType >  ChangeLabelFilterType;
  ChangeLabelFilterType::Pointer ChangeLabelFilter = ChangeLabelFilterType::New();

  ChangeLabelFilter->SetInput( RelabelFilter->GetOutput() );

  LabelListType::iterator iter;
  for (iter = labelsToRemove.begin(); iter != labelsToRemove.end(); iter ++)
    {
    if (*iter!=0)
      {
      //BinaryFilter->GetOutput()->RemoveLabel(*iter);
      ChangeLabelFilter->SetChange (*iter, 0);
      }
    }
  ChangeLabelFilter->Update();

  needleFilter->SetInput( ChangeLabelFilter->GetOutput() );
  needleFilter->SetMinPrincipalAxisLength( static_cast< float >(minPrincipalAxisLength) );
  needleFilter->SetMaxThickness (static_cast< float >(maxThickness) );
  needleFilter->SetAngleThreshold (static_cast< double >(anglethreshold) );

  // Set default orientation and closest point of the needle for detection
  // Note that the parameter is passed in RAS coordinate system
  // and must be converted to LPS coordinate system
  needleFilter->SetNormal (static_cast< double >(-normal[0]),
                           static_cast< double >(-normal[1]),
                           static_cast< double >(normal[2]));
  needleFilter->SetClosestPoint(static_cast< double >(-closestPoint[0]),
                                static_cast< double >(-closestPoint[1]),
                                static_cast< double >(closestPoint[2]));

  //writer->SetInput( needleFilter->GetOutput() );
  //writer->SetInput( RelabelFilter->GetOutput() );

  typedef   itk::ImageFileWriter< OutputImageType > WriterType;
  typename WriterType::Pointer writer = WriterType::New();

  writer->SetFileName( outputVolume.c_str() );
  writer->SetInput( needleFilter->GetOutput() );
  writer->SetUseCompression(1);

  try
    {
    writer->Update();
    }
  catch (itk::ExceptionObject &err)
    {
    std::cerr << err << std::endl;
    return EXIT_FAILURE ;
    }

  //typedef typename NeedleFilterType::NeedleTransformType TransformType;
  //TransformType::Pointer transform = needleFilter->GetNeedleTransform();

  //if (needleTransform != "")
  //  {
  //  typedef itk::TransformFileWriter TransformWriterType;
  //  TransformWriterType::Pointer needleTransformWriter;
  //  needleTransformWriter= TransformWriterType::New();
  //  needleTransformWriter->SetFileName( needleTransform );
  //  needleTransformWriter->SetInput( transform );
  //  try
  //    {
  //    needleTransformWriter->Update();
  //    }
  //  catch (itk::ExceptionObject &err)
  //    {
  //    std::cerr << err << std::endl;
  //    return EXIT_FAILURE ;
  //    }
  //  }

  return EXIT_SUCCESS;
}

} // end of anonymous namespace


int main( int argc, char * argv[] )
{
  PARSE_ARGS;

  itk::ImageIOBase::IOPixelType pixelType;
  itk::ImageIOBase::IOComponentType componentType;

  try
    {
    itk::GetImageType (inputVolume, pixelType, componentType);

    // This filter handles all types on input, but only produces
    // signed types
    switch (componentType)
      {
      case itk::ImageIOBase::UCHAR:
        return DoIt( argc, argv, static_cast<unsigned char>(0));
        break;
      case itk::ImageIOBase::CHAR:
        return DoIt( argc, argv, static_cast<char>(0));
        break;
      case itk::ImageIOBase::USHORT:
        return DoIt( argc, argv, static_cast<unsigned short>(0));
        break;
      case itk::ImageIOBase::SHORT:
        return DoIt( argc, argv, static_cast<short>(0));
        break;
      case itk::ImageIOBase::UINT:
        return DoIt( argc, argv, static_cast<unsigned int>(0));
        break;
      case itk::ImageIOBase::INT:
        return DoIt( argc, argv, static_cast<int>(0));
        break;
      case itk::ImageIOBase::ULONG:
        return DoIt( argc, argv, static_cast<unsigned long>(0));
        break;
      case itk::ImageIOBase::LONG:
        return DoIt( argc, argv, static_cast<long>(0));
        break;
      case itk::ImageIOBase::FLOAT:
        return DoIt( argc, argv, static_cast<float>(0));
        break;
      case itk::ImageIOBase::DOUBLE:
        return DoIt( argc, argv, static_cast<double>(0));
        break;
      case itk::ImageIOBase::UNKNOWNCOMPONENTTYPE:
      default:
        std::cout << "unknown component type" << std::endl;
        break;
      }
    }

  catch( itk::ExceptionObject &excep)
    {
    std::cerr << argv[0] << ": exception caught !" << std::endl;
    std::cerr << excep << std::endl;
    return EXIT_FAILURE;
    }
  return EXIT_SUCCESS;
}
