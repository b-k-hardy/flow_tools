#!/Applications/ParaView-5.12.0.app/Contents/bin pvpython
# trace generated using paraview version 5.12.0
# import paraview
# paraview.compatibility.major = 5
# paraview.compatibility.minor = 12

# NO IDEA IF THE INIT IS EVEN NECESSARY... PROBABLY

#### import the simple module from the paraview
from paraview.simple import *

#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# create a new 'XML Image Data Reader'
prab_flow_000vti = XMLImageDataReader(
    registrationName="Prab_flow_000.vti*",
    FileName=[
        "/Users/bkhardy/Dropbox (University of Michigan)/4D Flow Test Data/Prab 9.27.23/Prab_flow_vti/Prab_flow_000.vti",
        "/Users/bkhardy/Dropbox (University of Michigan)/4D Flow Test Data/Prab 9.27.23/Prab_flow_vti/Prab_flow_001.vti",
        "/Users/bkhardy/Dropbox (University of Michigan)/4D Flow Test Data/Prab 9.27.23/Prab_flow_vti/Prab_flow_002.vti",
        "/Users/bkhardy/Dropbox (University of Michigan)/4D Flow Test Data/Prab 9.27.23/Prab_flow_vti/Prab_flow_003.vti",
        "/Users/bkhardy/Dropbox (University of Michigan)/4D Flow Test Data/Prab 9.27.23/Prab_flow_vti/Prab_flow_004.vti",
        "/Users/bkhardy/Dropbox (University of Michigan)/4D Flow Test Data/Prab 9.27.23/Prab_flow_vti/Prab_flow_005.vti",
        "/Users/bkhardy/Dropbox (University of Michigan)/4D Flow Test Data/Prab 9.27.23/Prab_flow_vti/Prab_flow_006.vti",
        "/Users/bkhardy/Dropbox (University of Michigan)/4D Flow Test Data/Prab 9.27.23/Prab_flow_vti/Prab_flow_007.vti",
        "/Users/bkhardy/Dropbox (University of Michigan)/4D Flow Test Data/Prab 9.27.23/Prab_flow_vti/Prab_flow_008.vti",
        "/Users/bkhardy/Dropbox (University of Michigan)/4D Flow Test Data/Prab 9.27.23/Prab_flow_vti/Prab_flow_009.vti",
        "/Users/bkhardy/Dropbox (University of Michigan)/4D Flow Test Data/Prab 9.27.23/Prab_flow_vti/Prab_flow_010.vti",
        "/Users/bkhardy/Dropbox (University of Michigan)/4D Flow Test Data/Prab 9.27.23/Prab_flow_vti/Prab_flow_011.vti",
        "/Users/bkhardy/Dropbox (University of Michigan)/4D Flow Test Data/Prab 9.27.23/Prab_flow_vti/Prab_flow_012.vti",
        "/Users/bkhardy/Dropbox (University of Michigan)/4D Flow Test Data/Prab 9.27.23/Prab_flow_vti/Prab_flow_013.vti",
        "/Users/bkhardy/Dropbox (University of Michigan)/4D Flow Test Data/Prab 9.27.23/Prab_flow_vti/Prab_flow_014.vti",
        "/Users/bkhardy/Dropbox (University of Michigan)/4D Flow Test Data/Prab 9.27.23/Prab_flow_vti/Prab_flow_015.vti",
        "/Users/bkhardy/Dropbox (University of Michigan)/4D Flow Test Data/Prab 9.27.23/Prab_flow_vti/Prab_flow_016.vti",
        "/Users/bkhardy/Dropbox (University of Michigan)/4D Flow Test Data/Prab 9.27.23/Prab_flow_vti/Prab_flow_017.vti",
        "/Users/bkhardy/Dropbox (University of Michigan)/4D Flow Test Data/Prab 9.27.23/Prab_flow_vti/Prab_flow_018.vti",
        "/Users/bkhardy/Dropbox (University of Michigan)/4D Flow Test Data/Prab 9.27.23/Prab_flow_vti/Prab_flow_019.vti",
        "/Users/bkhardy/Dropbox (University of Michigan)/4D Flow Test Data/Prab 9.27.23/Prab_flow_vti/Prab_flow_020.vti",
        "/Users/bkhardy/Dropbox (University of Michigan)/4D Flow Test Data/Prab 9.27.23/Prab_flow_vti/Prab_flow_021.vti",
        "/Users/bkhardy/Dropbox (University of Michigan)/4D Flow Test Data/Prab 9.27.23/Prab_flow_vti/Prab_flow_022.vti",
        "/Users/bkhardy/Dropbox (University of Michigan)/4D Flow Test Data/Prab 9.27.23/Prab_flow_vti/Prab_flow_023.vti",
        "/Users/bkhardy/Dropbox (University of Michigan)/4D Flow Test Data/Prab 9.27.23/Prab_flow_vti/Prab_flow_024.vti",
        "/Users/bkhardy/Dropbox (University of Michigan)/4D Flow Test Data/Prab 9.27.23/Prab_flow_vti/Prab_flow_025.vti",
        "/Users/bkhardy/Dropbox (University of Michigan)/4D Flow Test Data/Prab 9.27.23/Prab_flow_vti/Prab_flow_026.vti",
        "/Users/bkhardy/Dropbox (University of Michigan)/4D Flow Test Data/Prab 9.27.23/Prab_flow_vti/Prab_flow_027.vti",
        "/Users/bkhardy/Dropbox (University of Michigan)/4D Flow Test Data/Prab 9.27.23/Prab_flow_vti/Prab_flow_028.vti",
        "/Users/bkhardy/Dropbox (University of Michigan)/4D Flow Test Data/Prab 9.27.23/Prab_flow_vti/Prab_flow_029.vti",
        "/Users/bkhardy/Dropbox (University of Michigan)/4D Flow Test Data/Prab 9.27.23/Prab_flow_vti/Prab_flow_030.vti",
    ],
)
prab_flow_000vti.CellArrayStatus = ["Velocity"]

# get animation scene
animationScene1 = GetAnimationScene()

# update animation scene based on data timesteps
animationScene1.UpdateAnimationUsingDataTimeSteps()

# Properties modified on prab_flow_000vti
prab_flow_000vti.TimeArray = "None"

# get active view
renderView1 = GetActiveViewOrCreate("RenderView")

# show data in view
prab_flow_000vtiDisplay = Show(
    prab_flow_000vti, renderView1, "UniformGridRepresentation"
)

# trace defaults for the display properties.
prab_flow_000vtiDisplay.Representation = "Outline"
prab_flow_000vtiDisplay.ColorArrayName = [None, ""]
prab_flow_000vtiDisplay.SelectTCoordArray = "None"
prab_flow_000vtiDisplay.SelectNormalArray = "None"
prab_flow_000vtiDisplay.SelectTangentArray = "None"
prab_flow_000vtiDisplay.OSPRayScaleArray = "Velocity"
prab_flow_000vtiDisplay.OSPRayScaleFunction = "Piecewise Function"
prab_flow_000vtiDisplay.Assembly = ""
prab_flow_000vtiDisplay.SelectOrientationVectors = "Velocity"
prab_flow_000vtiDisplay.ScaleFactor = 0.04200000000000001
prab_flow_000vtiDisplay.SelectScaleArray = "None"
prab_flow_000vtiDisplay.GlyphType = "Arrow"
prab_flow_000vtiDisplay.GlyphTableIndexArray = "None"
prab_flow_000vtiDisplay.GaussianRadius = 0.0021000000000000003
prab_flow_000vtiDisplay.SetScaleArray = ["POINTS", "Velocity"]
prab_flow_000vtiDisplay.ScaleTransferFunction = "Piecewise Function"
prab_flow_000vtiDisplay.OpacityArray = ["POINTS", "Velocity"]
prab_flow_000vtiDisplay.OpacityTransferFunction = "Piecewise Function"
prab_flow_000vtiDisplay.DataAxesGrid = "Grid Axes Representation"
prab_flow_000vtiDisplay.PolarAxes = "Polar Axes Representation"
prab_flow_000vtiDisplay.ScalarOpacityUnitDistance = 0.0052486676005553574
prab_flow_000vtiDisplay.OpacityArrayName = ["CELLS", "Velocity"]
prab_flow_000vtiDisplay.ColorArray2Name = ["CELLS", "Velocity"]
prab_flow_000vtiDisplay.SliceFunction = "Plane"
prab_flow_000vtiDisplay.Slice = 28
prab_flow_000vtiDisplay.SelectInputVectors = ["POINTS", "Velocity"]
prab_flow_000vtiDisplay.WriteLog = ""

# init the 'Piecewise Function' selected for 'ScaleTransferFunction'
prab_flow_000vtiDisplay.ScaleTransferFunction.Points = [
    -0.5029296875,
    0.0,
    0.5,
    0.0,
    0.5830078125,
    1.0,
    0.5,
    0.0,
]

# init the 'Piecewise Function' selected for 'OpacityTransferFunction'
prab_flow_000vtiDisplay.OpacityTransferFunction.Points = [
    -0.5029296875,
    0.0,
    0.5,
    0.0,
    0.5830078125,
    1.0,
    0.5,
    0.0,
]

# init the 'Plane' selected for 'SliceFunction'
prab_flow_000vtiDisplay.SliceFunction.Origin = [0.21000000000000002, 0.168, 0.056]

# reset view to fit data
renderView1.ResetCamera(False, 0.9)

# get the material library
materialLibrary1 = GetMaterialLibrary()

# update the view to ensure updated data information
renderView1.Update()

# create a new 'Threshold'
threshold1 = Threshold(registrationName="Threshold1", Input=prab_flow_000vti)
threshold1.Scalars = ["CELLS", "Velocity"]
threshold1.LowerThreshold = -0.5029296875
threshold1.UpperThreshold = 0.5830078125

# Properties modified on threshold1
threshold1.SelectedComponent = "Magnitude"
threshold1.UpperThreshold = 1e-16
threshold1.ThresholdMethod = "Above Upper Threshold"

# show data in view
threshold1Display = Show(threshold1, renderView1, "UnstructuredGridRepresentation")

# trace defaults for the display properties.
threshold1Display.Representation = "Surface"
threshold1Display.ColorArrayName = [None, ""]
threshold1Display.SelectTCoordArray = "None"
threshold1Display.SelectNormalArray = "None"
threshold1Display.SelectTangentArray = "None"
threshold1Display.OSPRayScaleArray = "Velocity"
threshold1Display.OSPRayScaleFunction = "Piecewise Function"
threshold1Display.Assembly = ""
threshold1Display.SelectOrientationVectors = "Velocity"
threshold1Display.ScaleFactor = 0.025725000351667405
threshold1Display.SelectScaleArray = "None"
threshold1Display.GlyphType = "Arrow"
threshold1Display.GlyphTableIndexArray = "None"
threshold1Display.GaussianRadius = 0.0012862500175833702
threshold1Display.SetScaleArray = ["POINTS", "Velocity"]
threshold1Display.ScaleTransferFunction = "Piecewise Function"
threshold1Display.OpacityArray = ["POINTS", "Velocity"]
threshold1Display.OpacityTransferFunction = "Piecewise Function"
threshold1Display.DataAxesGrid = "Grid Axes Representation"
threshold1Display.PolarAxes = "Polar Axes Representation"
threshold1Display.ScalarOpacityUnitDistance = 0.01481903253561648
threshold1Display.OpacityArrayName = ["CELLS", "Velocity"]
threshold1Display.SelectInputVectors = ["POINTS", "Velocity"]
threshold1Display.WriteLog = ""

# init the 'Piecewise Function' selected for 'ScaleTransferFunction'
threshold1Display.ScaleTransferFunction.Points = [
    -0.5029296875,
    0.0,
    0.5,
    0.0,
    0.5830078125,
    1.0,
    0.5,
    0.0,
]

# init the 'Piecewise Function' selected for 'OpacityTransferFunction'
threshold1Display.OpacityTransferFunction.Points = [
    -0.5029296875,
    0.0,
    0.5,
    0.0,
    0.5830078125,
    1.0,
    0.5,
    0.0,
]

# update the view to ensure updated data information
renderView1.Update()

# hide data in view
Hide(prab_flow_000vti, renderView1)

renderView1.ApplyIsometricView()

# reset view to fit data
renderView1.ResetCamera(False, 0.9)

renderView1.ResetActiveCameraToPositiveY()

# reset view to fit data
renderView1.ResetCamera(False, 0.9)

renderView1.ResetActiveCameraToNegativeY()

# reset view to fit data
renderView1.ResetCamera(False, 0.9)

renderView1.ResetActiveCameraToNegativeX()

# reset view to fit data
renderView1.ResetCamera(False, 0.9)

renderView1.ResetActiveCameraToPositiveX()

# reset view to fit data
renderView1.ResetCamera(False, 0.9)

renderView1.ResetActiveCameraToNegativeZ()

# reset view to fit data
renderView1.ResetCamera(False, 0.9)

renderView1.ResetActiveCameraToPositiveZ()

# reset view to fit data
renderView1.ResetCamera(False, 0.9)

renderView1.ApplyIsometricView()

# reset view to fit data
renderView1.ResetCamera(False, 0.9)

renderView1.AdjustRoll(-90.0)

renderView1.AdjustRoll(90.0)

renderView1.ResetActiveCameraToNegativeZ()

# reset view to fit data
renderView1.ResetCamera(False, 0.9)

renderView1.AdjustRoll(90.0)

renderView1.AdjustRoll(90.0)

renderView1.AdjustRoll(90.0)

renderView1.ResetActiveCameraToPositiveZ()

# reset view to fit data
renderView1.ResetCamera(False, 0.9)

renderView1.AdjustRoll(-90.0)

renderView1.AdjustRoll(-90.0)

renderView1.AdjustRoll(-90.0)

renderView1.ResetActiveCameraToPositiveZ()

# reset view to fit data
renderView1.ResetCamera(False, 0.9)

renderView1.AdjustRoll(-90.0)

renderView1.AdjustRoll(-90.0)

renderView1.AdjustRoll(-90.0)

# set scalar coloring
ColorBy(threshold1Display, ("CELLS", "Velocity", "Magnitude"))

# rescale color and/or opacity maps used to include current data range
threshold1Display.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
threshold1Display.SetScalarBarVisibility(renderView1, True)

# get 2D transfer function for 'Velocity'
velocityTF2D = GetTransferFunction2D("Velocity")

# get color transfer function/color map for 'Velocity'
velocityLUT = GetColorTransferFunction("Velocity")
velocityLUT.TransferFunction2D = velocityTF2D
velocityLUT.RGBPoints = [
    0.008513474499102879,
    0.231373,
    0.298039,
    0.752941,
    0.32418716749039583,
    0.865003,
    0.865003,
    0.865003,
    0.6398608604816888,
    0.705882,
    0.0156863,
    0.14902,
]
velocityLUT.ScalarRangeInitialized = 1.0

# get opacity transfer function/opacity map for 'Velocity'
velocityPWF = GetOpacityTransferFunction("Velocity")
velocityPWF.Points = [
    0.008513474499102879,
    0.0,
    0.5,
    0.0,
    0.6398608604816888,
    1.0,
    0.5,
    0.0,
]
velocityPWF.ScalarRangeInitialized = 1

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
velocityLUT.ApplyPreset("Black-Body Radiation", True)

# get layout
layout1 = GetLayout()

# split cell
layout1.SplitHorizontal(0, 0.5)

# set active view
SetActiveView(None)

# Create a new 'Render View'
renderView2 = CreateView("RenderView")
renderView2.AxesGrid = "Grid Axes 3D Actor"
renderView2.StereoType = "Crystal Eyes"
renderView2.CameraFocalDisk = 1.0
renderView2.LegendGrid = "Legend Grid Actor"
renderView2.BackEnd = "OSPRay raycaster"
renderView2.OSPRayMaterialLibrary = materialLibrary1

# assign view to a particular cell in the layout
AssignViewToLayout(view=renderView2, layout=layout1, hint=2)

# create a new 'XML Image Data Reader'
prab_p_STE_000vti = XMLImageDataReader(
    registrationName="Prab_p_STE_000.vti*",
    FileName=[
        "/Users/bkhardy/Dropbox (University of Michigan)/4D Flow Test Data/Prab 9.27.23/Prab_STE_vti/Prab_p_STE_000.vti",
        "/Users/bkhardy/Dropbox (University of Michigan)/4D Flow Test Data/Prab 9.27.23/Prab_STE_vti/Prab_p_STE_001.vti",
        "/Users/bkhardy/Dropbox (University of Michigan)/4D Flow Test Data/Prab 9.27.23/Prab_STE_vti/Prab_p_STE_002.vti",
        "/Users/bkhardy/Dropbox (University of Michigan)/4D Flow Test Data/Prab 9.27.23/Prab_STE_vti/Prab_p_STE_003.vti",
        "/Users/bkhardy/Dropbox (University of Michigan)/4D Flow Test Data/Prab 9.27.23/Prab_STE_vti/Prab_p_STE_004.vti",
        "/Users/bkhardy/Dropbox (University of Michigan)/4D Flow Test Data/Prab 9.27.23/Prab_STE_vti/Prab_p_STE_005.vti",
        "/Users/bkhardy/Dropbox (University of Michigan)/4D Flow Test Data/Prab 9.27.23/Prab_STE_vti/Prab_p_STE_006.vti",
        "/Users/bkhardy/Dropbox (University of Michigan)/4D Flow Test Data/Prab 9.27.23/Prab_STE_vti/Prab_p_STE_007.vti",
        "/Users/bkhardy/Dropbox (University of Michigan)/4D Flow Test Data/Prab 9.27.23/Prab_STE_vti/Prab_p_STE_008.vti",
        "/Users/bkhardy/Dropbox (University of Michigan)/4D Flow Test Data/Prab 9.27.23/Prab_STE_vti/Prab_p_STE_009.vti",
        "/Users/bkhardy/Dropbox (University of Michigan)/4D Flow Test Data/Prab 9.27.23/Prab_STE_vti/Prab_p_STE_010.vti",
        "/Users/bkhardy/Dropbox (University of Michigan)/4D Flow Test Data/Prab 9.27.23/Prab_STE_vti/Prab_p_STE_011.vti",
        "/Users/bkhardy/Dropbox (University of Michigan)/4D Flow Test Data/Prab 9.27.23/Prab_STE_vti/Prab_p_STE_012.vti",
        "/Users/bkhardy/Dropbox (University of Michigan)/4D Flow Test Data/Prab 9.27.23/Prab_STE_vti/Prab_p_STE_013.vti",
        "/Users/bkhardy/Dropbox (University of Michigan)/4D Flow Test Data/Prab 9.27.23/Prab_STE_vti/Prab_p_STE_014.vti",
        "/Users/bkhardy/Dropbox (University of Michigan)/4D Flow Test Data/Prab 9.27.23/Prab_STE_vti/Prab_p_STE_015.vti",
        "/Users/bkhardy/Dropbox (University of Michigan)/4D Flow Test Data/Prab 9.27.23/Prab_STE_vti/Prab_p_STE_016.vti",
        "/Users/bkhardy/Dropbox (University of Michigan)/4D Flow Test Data/Prab 9.27.23/Prab_STE_vti/Prab_p_STE_017.vti",
        "/Users/bkhardy/Dropbox (University of Michigan)/4D Flow Test Data/Prab 9.27.23/Prab_STE_vti/Prab_p_STE_018.vti",
        "/Users/bkhardy/Dropbox (University of Michigan)/4D Flow Test Data/Prab 9.27.23/Prab_STE_vti/Prab_p_STE_019.vti",
        "/Users/bkhardy/Dropbox (University of Michigan)/4D Flow Test Data/Prab 9.27.23/Prab_STE_vti/Prab_p_STE_020.vti",
        "/Users/bkhardy/Dropbox (University of Michigan)/4D Flow Test Data/Prab 9.27.23/Prab_STE_vti/Prab_p_STE_021.vti",
        "/Users/bkhardy/Dropbox (University of Michigan)/4D Flow Test Data/Prab 9.27.23/Prab_STE_vti/Prab_p_STE_022.vti",
        "/Users/bkhardy/Dropbox (University of Michigan)/4D Flow Test Data/Prab 9.27.23/Prab_STE_vti/Prab_p_STE_023.vti",
        "/Users/bkhardy/Dropbox (University of Michigan)/4D Flow Test Data/Prab 9.27.23/Prab_STE_vti/Prab_p_STE_024.vti",
        "/Users/bkhardy/Dropbox (University of Michigan)/4D Flow Test Data/Prab 9.27.23/Prab_STE_vti/Prab_p_STE_025.vti",
        "/Users/bkhardy/Dropbox (University of Michigan)/4D Flow Test Data/Prab 9.27.23/Prab_STE_vti/Prab_p_STE_026.vti",
        "/Users/bkhardy/Dropbox (University of Michigan)/4D Flow Test Data/Prab 9.27.23/Prab_STE_vti/Prab_p_STE_027.vti",
        "/Users/bkhardy/Dropbox (University of Michigan)/4D Flow Test Data/Prab 9.27.23/Prab_STE_vti/Prab_p_STE_028.vti",
        "/Users/bkhardy/Dropbox (University of Michigan)/4D Flow Test Data/Prab 9.27.23/Prab_STE_vti/Prab_p_STE_029.vti",
    ],
)
prab_p_STE_000vti.CellArrayStatus = ["Pressure"]

# Properties modified on prab_p_STE_000vti
prab_p_STE_000vti.TimeArray = "None"

# show data in view
prab_p_STE_000vtiDisplay = Show(
    prab_p_STE_000vti, renderView2, "UniformGridRepresentation"
)

# trace defaults for the display properties.
prab_p_STE_000vtiDisplay.Representation = "Outline"
prab_p_STE_000vtiDisplay.ColorArrayName = ["CELLS", ""]
prab_p_STE_000vtiDisplay.SelectTCoordArray = "None"
prab_p_STE_000vtiDisplay.SelectNormalArray = "None"
prab_p_STE_000vtiDisplay.SelectTangentArray = "None"
prab_p_STE_000vtiDisplay.OSPRayScaleArray = "Pressure"
prab_p_STE_000vtiDisplay.OSPRayScaleFunction = "Piecewise Function"
prab_p_STE_000vtiDisplay.Assembly = ""
prab_p_STE_000vtiDisplay.SelectOrientationVectors = "None"
prab_p_STE_000vtiDisplay.ScaleFactor = 0.04252500000000001
prab_p_STE_000vtiDisplay.SelectScaleArray = "Pressure"
prab_p_STE_000vtiDisplay.GlyphType = "Arrow"
prab_p_STE_000vtiDisplay.GlyphTableIndexArray = "Pressure"
prab_p_STE_000vtiDisplay.GaussianRadius = 0.00212625
prab_p_STE_000vtiDisplay.SetScaleArray = ["POINTS", "Pressure"]
prab_p_STE_000vtiDisplay.ScaleTransferFunction = "Piecewise Function"
prab_p_STE_000vtiDisplay.OpacityArray = ["POINTS", "Pressure"]
prab_p_STE_000vtiDisplay.OpacityTransferFunction = "Piecewise Function"
prab_p_STE_000vtiDisplay.DataAxesGrid = "Grid Axes Representation"
prab_p_STE_000vtiDisplay.PolarAxes = "Polar Axes Representation"
prab_p_STE_000vtiDisplay.ScalarOpacityUnitDistance = 0.0026074143193960567
prab_p_STE_000vtiDisplay.OpacityArrayName = ["CELLS", "Pressure"]
prab_p_STE_000vtiDisplay.ColorArray2Name = ["CELLS", "Pressure"]
prab_p_STE_000vtiDisplay.IsosurfaceValues = [0.6893318007300475]
prab_p_STE_000vtiDisplay.SliceFunction = "Plane"
prab_p_STE_000vtiDisplay.Slice = 58
prab_p_STE_000vtiDisplay.SelectInputVectors = [None, ""]
prab_p_STE_000vtiDisplay.WriteLog = ""

# init the 'Piecewise Function' selected for 'ScaleTransferFunction'
prab_p_STE_000vtiDisplay.ScaleTransferFunction.Points = [
    -1.9337503982804463,
    0.0,
    0.5,
    0.0,
    3.312413999740541,
    1.0,
    0.5,
    0.0,
]

# init the 'Piecewise Function' selected for 'OpacityTransferFunction'
prab_p_STE_000vtiDisplay.OpacityTransferFunction.Points = [
    -1.9337503982804463,
    0.0,
    0.5,
    0.0,
    3.312413999740541,
    1.0,
    0.5,
    0.0,
]

# init the 'Plane' selected for 'SliceFunction'
prab_p_STE_000vtiDisplay.SliceFunction.Origin = [0.212625, 0.170625, 0.058]

# reset view to fit data
renderView2.ResetCamera(False, 0.9)

# update the view to ensure updated data information
renderView2.Update()

# create a new 'Threshold'
threshold2 = Threshold(registrationName="Threshold2", Input=prab_p_STE_000vti)
threshold2.Scalars = ["CELLS", "Pressure"]
threshold2.LowerThreshold = -1.9337503982804463
threshold2.UpperThreshold = 3.3124139997405413

# Properties modified on threshold2
threshold2.LowerThreshold = 0.0
threshold2.UpperThreshold = 0.0
threshold2.Invert = 1

# show data in view
threshold2Display = Show(threshold2, renderView2, "UnstructuredGridRepresentation")

# get 2D transfer function for 'Pressure'
pressureTF2D = GetTransferFunction2D("Pressure")

# get color transfer function/color map for 'Pressure'
pressureLUT = GetColorTransferFunction("Pressure")
pressureLUT.TransferFunction2D = pressureTF2D
pressureLUT.RGBPoints = [
    -1.9337503982804463,
    0.231373,
    0.298039,
    0.752941,
    0.6893318007300473,
    0.865003,
    0.865003,
    0.865003,
    3.312413999740541,
    0.705882,
    0.0156863,
    0.14902,
]
pressureLUT.ScalarRangeInitialized = 1.0

# get opacity transfer function/opacity map for 'Pressure'
pressurePWF = GetOpacityTransferFunction("Pressure")
pressurePWF.Points = [
    -1.9337503982804463,
    0.0,
    0.5,
    0.0,
    3.312413999740541,
    1.0,
    0.5,
    0.0,
]
pressurePWF.ScalarRangeInitialized = 1

# trace defaults for the display properties.
threshold2Display.Representation = "Surface"
threshold2Display.ColorArrayName = ["CELLS", "Pressure"]
threshold2Display.LookupTable = pressureLUT
threshold2Display.SelectTCoordArray = "None"
threshold2Display.SelectNormalArray = "None"
threshold2Display.SelectTangentArray = "None"
threshold2Display.OSPRayScaleArray = "Pressure"
threshold2Display.OSPRayScaleFunction = "Piecewise Function"
threshold2Display.Assembly = ""
threshold2Display.SelectOrientationVectors = "None"
threshold2Display.ScaleFactor = 0.02296874970197678
threshold2Display.SelectScaleArray = "Pressure"
threshold2Display.GlyphType = "Arrow"
threshold2Display.GlyphTableIndexArray = "Pressure"
threshold2Display.GaussianRadius = 0.001148437485098839
threshold2Display.SetScaleArray = ["POINTS", "Pressure"]
threshold2Display.ScaleTransferFunction = "Piecewise Function"
threshold2Display.OpacityArray = ["POINTS", "Pressure"]
threshold2Display.OpacityTransferFunction = "Piecewise Function"
threshold2Display.DataAxesGrid = "Grid Axes Representation"
threshold2Display.PolarAxes = "Polar Axes Representation"
threshold2Display.ScalarOpacityFunction = pressurePWF
threshold2Display.ScalarOpacityUnitDistance = 0.007336283647051825
threshold2Display.OpacityArrayName = ["CELLS", "Pressure"]
threshold2Display.SelectInputVectors = [None, ""]
threshold2Display.WriteLog = ""

# init the 'Piecewise Function' selected for 'ScaleTransferFunction'
threshold2Display.ScaleTransferFunction.Points = [
    -1.9337503982804463,
    0.0,
    0.5,
    0.0,
    3.312413999740541,
    1.0,
    0.5,
    0.0,
]

# init the 'Piecewise Function' selected for 'OpacityTransferFunction'
threshold2Display.OpacityTransferFunction.Points = [
    -1.9337503982804463,
    0.0,
    0.5,
    0.0,
    3.312413999740541,
    1.0,
    0.5,
    0.0,
]

# show color bar/color legend
threshold2Display.SetScalarBarVisibility(renderView2, True)

# update the view to ensure updated data information
renderView2.Update()

# Rescale transfer function
pressureLUT.RescaleTransferFunction(-1.9337503982804463, 3.3124139997405413)

# Rescale transfer function
pressurePWF.RescaleTransferFunction(-1.9337503982804463, 3.3124139997405413)

# hide data in view
Hide(prab_p_STE_000vti, renderView2)

renderView2.ResetActiveCameraToPositiveZ()

# reset view to fit data
renderView2.ResetCamera(False, 0.9)

renderView2.AdjustRoll(-90.0)

renderView2.AdjustRoll(-90.0)

renderView2.AdjustRoll(-90.0)

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
pressureLUT.ApplyPreset("Cool to Warm (Extended)", True)

# set active view
SetActiveView(renderView1)

# Hide orientation axes
renderView1.OrientationAxesVisibility = 0

# set active view
SetActiveView(renderView2)

# Hide orientation axes
renderView2.OrientationAxesVisibility = 0

# set active view
SetActiveView(renderView1)

# get color legend/bar for velocityLUT in view renderView1
velocityLUTColorBar = GetScalarBar(velocityLUT, renderView1)
velocityLUTColorBar.Title = "Velocity"
velocityLUTColorBar.ComponentTitle = "Magnitude"

# change scalar bar placement
velocityLUTColorBar.Position = [0.8940750736015701, 0.007005899705014749]

# set active source
SetActiveSource(threshold1)

# Properties modified on velocityLUTColorBar
velocityLUTColorBar.Title = "|V|"
velocityLUTColorBar.ComponentTitle = ""

# Properties modified on velocityLUTColorBar
velocityLUTColorBar.Title = "||V||"

# Properties modified on velocityLUTColorBar
velocityLUTColorBar.Title = "/|V/|"

# Properties modified on velocityLUTColorBar
velocityLUTColorBar.Title = "Velocity"
velocityLUTColorBar.ComponentTitle = "Magnitude"

# Properties modified on velocityLUTColorBar
velocityLUTColorBar.ScalarBarThickness = 26

# Properties modified on velocityLUTColorBar
velocityLUTColorBar.ScalarBarThickness = 40

# Properties modified on velocityLUTColorBar
velocityLUTColorBar.ScalarBarLength = 0.4

# Properties modified on velocityLUTColorBar
velocityLUTColorBar.TitleFontSize = 30

# Properties modified on velocityLUTColorBar
velocityLUTColorBar.TitleBold = 1

# Properties modified on velocityLUTColorBar
velocityLUTColorBar.ScalarBarLength = 0.2

# Properties modified on velocityLUTColorBar
velocityLUTColorBar.ScalarBarLength = 0.3

# Properties modified on velocityLUTColorBar
velocityLUTColorBar.ScalarBarLength = 0.4

# Properties modified on velocityLUTColorBar
velocityLUTColorBar.RangeLabelFormat = "%-#6.1f"

# Properties modified on velocityLUTColorBar
velocityLUTColorBar.RangeLabelFormat = "%-#6.2f"

# Properties modified on velocityLUTColorBar
velocityLUTColorBar.AutomaticLabelFormat = 0
velocityLUTColorBar.LabelFormat = "%-#6.3f"

# Properties modified on velocityLUTColorBar
velocityLUTColorBar.LabelFormat = "%-#6.2f"

# Properties modified on velocityLUTColorBar
velocityLUTColorBar.DrawTickLabels = 0

# Properties modified on velocityLUTColorBar
velocityLUTColorBar.DrawTickLabels = 1

# Properties modified on velocityLUTColorBar
velocityLUTColorBar.LabelFontSize = 30

# Properties modified on velocityLUTColorBar
velocityLUTColorBar.DrawTickLabels = 0

# Properties modified on velocityLUTColorBar
velocityLUTColorBar.DrawTickLabels = 1

# Properties modified on velocityLUTColorBar
velocityLUTColorBar.AutomaticAnnotations = 1

# Properties modified on velocityLUTColorBar
velocityLUTColorBar.AutomaticAnnotations = 0

# Properties modified on velocityLUTColorBar
velocityLUTColorBar.DrawAnnotations = 0

# Properties modified on velocityLUTColorBar
velocityLUTColorBar.DrawAnnotations = 1

# Properties modified on velocityLUTColorBar
velocityLUTColorBar.LabelOpacity = 0.7000000000000001

# Properties modified on velocityLUTColorBar
velocityLUTColorBar.LabelOpacity = 1.0

# Properties modified on velocityLUTColorBar
velocityLUTColorBar.LabelFontSize = 16

# change scalar bar placement
velocityLUTColorBar.WindowLocation = "Any Location"
velocityLUTColorBar.Position = [0.5918179587831208, 0.2297197640117994]
velocityLUTColorBar.ScalarBarLength = 0.40000000000000036

# set active view
SetActiveView(renderView2)

# get color legend/bar for pressureLUT in view renderView2
pressureLUTColorBar = GetScalarBar(pressureLUT, renderView2)
pressureLUTColorBar.Title = "Pressure"
pressureLUTColorBar.ComponentTitle = ""

# change scalar bar placement
pressureLUTColorBar.WindowLocation = "Any Location"
pressureLUTColorBar.Position = [0.6080103042198234, 0.2643805309734514]
pressureLUTColorBar.ScalarBarLength = 0.3300000000000002

# reset view to fit data
renderView2.ResetCamera(True, 0.9)

# reset view to fit data
renderView2.ResetCamera(True, 0.9)

# set active view
SetActiveView(renderView1)

# reset view to fit data
renderView1.ResetCamera(True, 0.9)

# set active view
SetActiveView(renderView2)

# set active source
SetActiveSource(threshold2)

# Properties modified on pressureLUTColorBar
pressureLUTColorBar.TitleBold = 1
pressureLUTColorBar.TitleFontSize = 30

# Properties modified on pressureLUTColorBar
pressureLUTColorBar.ScalarBarThickness = 30
pressureLUTColorBar.ScalarBarLength = 0.4

# Properties modified on pressureLUTColorBar
pressureLUTColorBar.ScalarBarThickness = 40

# Properties modified on pressureLUTColorBar
pressureLUTColorBar.WindowLocation = "Lower Right Corner"

# Properties modified on pressureLUTColorBar
pressureLUTColorBar.AutomaticLabelFormat = 0
pressureLUTColorBar.LabelFormat = "%-#6.2f"
pressureLUTColorBar.RangeLabelFormat = "%-#6.2f"

# set active source
SetActiveSource(threshold1)

# set active view
SetActiveView(renderView1)

# reset view to fit data
renderView1.ResetCamera(True, 0.9)

# Properties modified on velocityLUTColorBar
velocityLUTColorBar.WindowLocation = "Lower Right Corner"

# create a new 'Glyph'
glyph1 = Glyph(registrationName="Glyph1", Input=threshold1, GlyphType="Arrow")
glyph1.OrientationArray = ["CELLS", "Velocity"]
glyph1.ScaleArray = ["POINTS", "No scale array"]
glyph1.ScaleFactor = 0.025725000351667405
glyph1.GlyphTransform = "Transform2"

# Properties modified on glyph1
glyph1.GlyphMode = "Every Nth Point"
glyph1.Stride = 2

# show data in view
glyph1Display = Show(glyph1, renderView1, "GeometryRepresentation")

# trace defaults for the display properties.
glyph1Display.Representation = "Surface"
glyph1Display.ColorArrayName = [None, ""]
glyph1Display.SelectTCoordArray = "None"
glyph1Display.SelectNormalArray = "None"
glyph1Display.SelectTangentArray = "None"
glyph1Display.OSPRayScaleArray = "Velocity"
glyph1Display.OSPRayScaleFunction = "Piecewise Function"
glyph1Display.Assembly = ""
glyph1Display.SelectOrientationVectors = "Velocity"
glyph1Display.ScaleFactor = 0.030295968800783158
glyph1Display.SelectScaleArray = "None"
glyph1Display.GlyphType = "Arrow"
glyph1Display.GlyphTableIndexArray = "None"
glyph1Display.GaussianRadius = 0.0015147984400391578
glyph1Display.SetScaleArray = ["POINTS", "Velocity"]
glyph1Display.ScaleTransferFunction = "Piecewise Function"
glyph1Display.OpacityArray = ["POINTS", "Velocity"]
glyph1Display.OpacityTransferFunction = "Piecewise Function"
glyph1Display.DataAxesGrid = "Grid Axes Representation"
glyph1Display.PolarAxes = "Polar Axes Representation"
glyph1Display.SelectInputVectors = ["POINTS", "Velocity"]
glyph1Display.WriteLog = ""

# init the 'Piecewise Function' selected for 'ScaleTransferFunction'
glyph1Display.ScaleTransferFunction.Points = [
    -0.5029296875,
    0.0,
    0.5,
    0.0,
    0.525390625,
    1.0,
    0.5,
    0.0,
]

# init the 'Piecewise Function' selected for 'OpacityTransferFunction'
glyph1Display.OpacityTransferFunction.Points = [
    -0.5029296875,
    0.0,
    0.5,
    0.0,
    0.525390625,
    1.0,
    0.5,
    0.0,
]

# update the view to ensure updated data information
renderView1.Update()

# Rescale transfer function
pressureLUT.RescaleTransferFunction(-1.9337503982804463, 3.3124139997405413)

# Rescale transfer function
pressurePWF.RescaleTransferFunction(-1.9337503982804463, 3.3124139997405413)

# set scalar coloring
ColorBy(glyph1Display, ("POINTS", "Velocity", "Magnitude"))

# rescale color and/or opacity maps used to include current data range
glyph1Display.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
glyph1Display.SetScalarBarVisibility(renderView1, True)

# hide data in view
Hide(threshold1, renderView1)

# set active source
SetActiveSource(threshold1)

# Rescale transfer function
velocityLUT.RescaleTransferFunction(0.0021836601342771385, 1.601406182825)

# Rescale transfer function
velocityPWF.RescaleTransferFunction(0.0021836601342771385, 1.601406182825)

# set active source
SetActiveSource(glyph1)

# Properties modified on glyph1
glyph1.ScaleArray = ["CELLS", "Velocity"]

# update the view to ensure updated data information
renderView1.Update()

# Rescale transfer function
pressureLUT.RescaleTransferFunction(-1.9337503982804463, 3.3124139997405413)

# Rescale transfer function
pressurePWF.RescaleTransferFunction(-1.9337503982804463, 3.3124139997405413)

# set active view
SetActiveView(renderView2)

# set active source
SetActiveSource(threshold2)

# change representation type
threshold2Display.SetRepresentationType("Volume")

# set active view
SetActiveView(renderView1)

LoadPalette(paletteName="WhiteBackground")

# set active view
SetActiveView(renderView2)

animationScene1.Play()

# Rescale transfer function
pressureLUT.RescaleTransferFunction(-16.200632834740205, 28.56982184639022)

# Rescale transfer function
pressurePWF.RescaleTransferFunction(-16.200632834740205, 28.56982184639022)

# set active source
SetActiveSource(glyph1)

# Properties modified on glyph1
glyph1.Stride = 3

# update the view to ensure updated data information
renderView1.Update()

# set active source
SetActiveSource(threshold2)

animationScene1.GoToFirst()

animationScene1.Play()

animationScene1.GoToFirst()

animationScene1.Play()

animationScene1.GoToFirst()

# layout/tab size in pixels
layout1.SetSize(4075, 2712)

# current camera placement for renderView1
renderView1.CameraPosition = [
    0.22297708261492463,
    0.17426923781609696,
    -0.6260659752459817,
]
renderView1.CameraFocalPoint = [
    0.23887500539422035,
    0.1785000041127205,
    0.05999999865889549,
]
renderView1.CameraViewUp = [
    -0.9997316334996834,
    0.00011427684375142638,
    0.023165662538719222,
]
renderView1.CameraViewAngle = 23.91838741396263
renderView1.CameraParallelScale = 0.18026557131471707

# current camera placement for renderView2
renderView2.CameraPosition = [
    0.22485067769382175,
    0.18046874552965164,
    -0.557465623158138,
]
renderView2.CameraFocalPoint = [
    0.227718748152256,
    0.18046874552965164,
    0.06149999983608723,
]
renderView2.CameraViewUp = [
    -0.9999892648137173,
    -6.661338147750939e-16,
    0.0046336009022432895,
]
renderView2.CameraViewAngle = 23.697148475909536
renderView2.CameraParallelScale = 0.1625897925890296

# save animation
SaveAnimation(
    filename="/Users/bkhardy/Dropbox (University of Michigan)/4D Flow Test Data/Prab 9.27.23/vid_attempt.avi",
    viewOrLayout=layout1,
    location=16,
    SaveAllViews=1,
    ImageResolution=[4076, 2712],
    FrameRate=30,
    FrameWindow=[0, 30],
    # FFMPEG options
    Compression=2,
)

# ================================================================
# addendum: following script captures some of the application
# state to faithfully reproduce the visualization during playback
# ================================================================

# --------------------------------
# saving layout sizes for layouts

# layout/tab size in pixels
layout1.SetSize(4077, 2712)

# -----------------------------------
# saving camera placements for views

# current camera placement for renderView1
renderView1.CameraPosition = [
    0.22297708261492463,
    0.17426923781609696,
    -0.6260659752459817,
]
renderView1.CameraFocalPoint = [
    0.23887500539422035,
    0.1785000041127205,
    0.05999999865889549,
]
renderView1.CameraViewUp = [
    -0.9997316334996834,
    0.00011427684375142638,
    0.023165662538719222,
]
renderView1.CameraViewAngle = 23.91838741396263
renderView1.CameraParallelScale = 0.18026557131471707

# current camera placement for renderView2
renderView2.CameraPosition = [
    0.22485067769382175,
    0.18046874552965164,
    -0.557465623158138,
]
renderView2.CameraFocalPoint = [
    0.227718748152256,
    0.18046874552965164,
    0.06149999983608723,
]
renderView2.CameraViewUp = [
    -0.9999892648137173,
    -6.661338147750939e-16,
    0.0046336009022432895,
]
renderView2.CameraViewAngle = 23.697148475909536
renderView2.CameraParallelScale = 0.1625897925890296


##--------------------------------------------
## You may need to add some code at the end of this python script depending on your usage, eg:
#
## Render all views to see them appears
# RenderAllViews()
#
## Interact with the view, usefull when running from pvpython
# Interact()
#
## Save a screenshot of the active view
# SaveScreenshot("path/to/screenshot.png")
#
## Save a screenshot of a layout (multiple splitted view)
# SaveScreenshot("path/to/screenshot.png", GetLayout())
#
## Save all "Extractors" from the pipeline browser
# SaveExtracts()
#
## Save a animation of the current active view
# SaveAnimation()
#
## Please refer to the documentation of paraview.simple
## https://kitware.github.io/paraview-docs/latest/python/paraview.simple.html
##--------------------------------------------
