public static final int JsPrivateCreator = 0x4A530010;               //String, "JiShi"

public static final int JsStationUID = 0x4A531001;                   //String, UI
public static final int JsStationGUID = 0x4A531002;                  //Long String, LO, 36 chars. Like '6F9619FF-8B86-D011-B42D-00C04FC964FF'

public static final int JsTransducerSN = 0x4A531005;                 //String, LO, <= 64 chars
public static final int JsTransducerGUID = 0x4A531006;               //Long String, LO, 36 chars.

public static final int JsCenterFrequency = 0x4A531010;              //Float, FL, in MHz
public static final int JsRenderFPS = 0x4A531020;                    //Float, FL
public static final int JsBModeGain = 0x4A531030;                    //Integer, US
public static final int JsDynamicRange = 0x4A531032;                 //Integer, US
public static final int JSAspectMode = 0x4A531040;                   //Integer, US

public static final int JSGrayscaleMap = 0x4A531060;                 //Integer Array, SL. Grayscale Map

public static final int JSAnnotationSequence = 0x4A531070;           //Annotation Sequence, SQ, Like Graphic Annotation Module

//From DICOM PS3.3 C.10.5 Graphic Annotation Module
public static final int NumberOfGraphicPoints = 0x00700022;          //Number of Graphic Points, US
public static final int GraphicData = 0x00700023;                    //Graphic Data, FL, 2-n
public static final int GraphicType = 0x00700023;                    //Graphic Type, CS


public static final int JSPWResult = 0x4A531090;                     //PW Result. FL, 0-n
public static final int JSPWEnvelope = 0x4A531091;                   //PW Envelope, SS, 0-n
public static final int JSPWSpectralSize = 0x4A531092;               //PW Length & LineCount of Spectral Data, US, 2
public static final int JSPWSpectralData = 0x4A531093;               //PW SpectralData, OB, (LineCount * Length)

public static final int JSStudyAttachment = 0x4A5310A0;              //Study Attachment. Binary, OB
public static final int JSApplicationKey = 0x4A5310A2;               //Application Key. Long String, LO

public static final int JSIMThickness = 0x4A5310A4;                  //IMT Mean/Min/Max in mm, FL, 3*NumberOfFrames

public static final int JSAdmittingDiagnoses = 0x4A5310C0;           //初诊意见, LT
