package mine.design.en.global;

import java.util.ArrayList;
import java.util.List;

public class Contant {
    //    模型
    public static final String Model1="VGG16";
    public static final String Model2="VGG19";
    public static final String Model3="ResNet50";
    //因为在选择模型和分类
    public static List<String> getModels(){
        List<String> list=new ArrayList<String>();
        list.add(Model1);
        list.add(Model2);
        list.add(Model3);
        return list;
    }
    //数据集
    public static final String Class1="RS_C11_Database";
    public static final String Class2="RSSCN7";
    public static final String Class3="SIRI WHU";
    //费用类别
    public static List<String> getDataMarket(){
        List<String> list=new ArrayList<String>();
        list.add(Class1);
        list.add(Class2);
        list.add(Class3);
        return list;
    }





}
